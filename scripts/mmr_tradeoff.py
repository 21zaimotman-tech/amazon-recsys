"""MMR bonus: tune lambda and measure the relevance-diversity trade-off.

For a sample of test users, run the full serving pipeline (two-tower retrieve
-> LightGBM rank) and re-rank the candidates with Maximal Marginal Relevance
at several lambda values. Report NDCG@10 (relevance) and two diversity views:
catalog coverage@10 across users, and mean intra-list embedding similarity
(lower = more diverse list). Writes data/mmr_tradeoff.csv for ANALYSIS.md.

Run AFTER notebook 05 has exported artifacts/ (uses the same files the API
serves). CPU-light: defaults to 2,000 sampled users.

    python -m scripts.mmr_tradeoff --users 2000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src import config as C
from src.data.split import ground_truth_from
from src.eval.metrics import evaluate
from src.eval.harness import seen_items, two_tower_user_vecs
from src.models.two_tower import TwoTower
from src.ranking.features import featurize, FEATURE_NAMES

DATA = C.DATA
ART = C.ARTIFACTS
LAMBDAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def mmr_order(cand_idx, rel, item_emb, k, lam):
    V = item_emb[cand_idx]
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    r = rel.astype("float64")
    if r.max() > r.min():
        r = (r - r.min()) / (r.max() - r.min())
    sims = V @ V.T
    selected, remaining = [], list(range(len(cand_idx)))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: r[i])
        else:
            best = max(remaining, key=lambda i: lam * r[i] - (1 - lam) * sims[i, selected].max())
        selected.append(best)
        remaining.remove(best)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    import pickle, torch, faiss
    faiss.omp_set_num_threads(1)

    train = pd.read_parquet(DATA / "train.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    items_meta = pd.read_parquet(DATA / "items.parquet")
    ids = json.load(open(DATA / "id_encoders.json"))
    item_ids = ids["item_ids"]
    uid2u = {u: i for i, u in enumerate(ids["user_ids"])}
    n_items = len(item_ids)

    ckpt = torch.load(DATA / "two_tower_checkpoint.pt", map_location="cpu")
    tt = TwoTower(ckpt["n_items_total"], ckpt["dim"], temperature=ckpt["temperature"])
    tt.load_state_dict(ckpt["state_dict"]); tt.eval()
    item_emb = np.load(DATA / "item_emb.npy")
    lgbm = pickle.load(open(ART / "lgbm.pkl", "rb"))
    fs = pickle.load(open(ART / "feature_store.pkl", "rb"))
    store, now_ts = fs["store"], fs["now_ts"]

    test_gt = ground_truth_from(test, positive_only=True)
    rng = np.random.default_rng(C.SEED)
    users = [u for u in test_gt if u in uid2u]
    users = list(rng.choice(users, size=min(args.users, len(users)), replace=False))

    hist_by_u = train.sort_values("timestamp").groupby("u")["i"].agg(list).to_dict()
    histories = [hist_by_u.get(uid2u[u], []) for u in users]
    uvecs = two_tower_user_vecs(tt, histories, max_hist=ckpt["max_hist"], device="cpu")
    emb = item_emb.astype("float32").copy(); faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
    q = uvecs.astype("float32"); faiss.normalize_L2(q)
    D, I = index.search(q, 100)

    seen = seen_items(train)
    id_of = np.asarray(item_ids)

    # LightGBM scores for every (user, candidate) in one batch
    rows_u, rows_i, rows_s = [], [], []
    per_user = []
    for r, u in enumerate(users):
        s = seen.get(u, set())
        kept = [(int(j), float(D[r, c])) for c, j in enumerate(I[r]) if id_of[j] not in s][:100]
        per_user.append(kept)
        for j, sc in kept:
            rows_u.append(u); rows_i.append(id_of[j]); rows_s.append(sc)
    pairs = pd.DataFrame({"user_id": rows_u, "item_id": rows_i, "retriever_score": rows_s})
    feats = featurize(pairs, store, now_ts)
    scores = lgbm.predict(feats[FEATURE_NAMES], num_threads=1)

    out = []
    pos = 0
    scored_per_user = []
    for kept in per_user:
        k = len(kept)
        scored_per_user.append((np.array([j for j, _ in kept]),
                                scores[pos:pos + k]))
        pos += k

    for lam in LAMBDAS:
        recs, ils = {}, []
        for u, (cand_idx, rel) in zip(users, scored_per_user):
            if len(cand_idx) == 0:
                continue
            sel = mmr_order(cand_idx, rel, item_emb, args.k, lam)
            chosen = cand_idx[sel]
            recs[u] = [id_of[j] for j in chosen]
            V = item_emb[chosen]
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
            S = V @ V.T
            iu = np.triu_indices(len(chosen), k=1)
            if len(iu[0]):
                ils.append(float(S[iu].mean()))
        m = evaluate(recs, test_gt, catalog_size=n_items, k_values=(args.k,))
        out.append({"lambda": lam,
                    f"NDCG@{args.k}": m[f"NDCG@{args.k}"],
                    f"Recall@{args.k}": m[f"Recall@{args.k}"],
                    f"Coverage@{args.k}": m[f"Coverage@{args.k}"],
                    "intra_list_similarity": float(np.mean(ils))})
        print(out[-1])

    df = pd.DataFrame(out)
    df.to_csv(DATA / "mmr_tradeoff.csv", index=False)
    print(f"saved -> {DATA / 'mmr_tradeoff.csv'}")


if __name__ == "__main__":
    main()
