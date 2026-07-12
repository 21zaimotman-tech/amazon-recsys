"""Online inference + per-component latency logging.

Pipeline (logged-in user):
  DB history -> user tower (TorchScript) -> FAISS top-100 -> LightGBM re-rank -> top-10
Cold/unknown users -> popularity. Missing artifact -> graceful degrade, so
`docker compose up` works from S2 (popularity only) onward.
"""
import time, json, pickle
from pathlib import Path
import numpy as np

ART = Path("/artifacts")          # mounted from ./artifacts


class Recommender:
    def __init__(self):
        self._load()

    def _load(self):
        self.popularity = self._maybe(lambda: json.load(open(ART / "popularity.json")))
        self.item_ids = self._maybe(lambda: json.load(open(ART / "item_encoder.json")))
        self.idx_map = {v: i for i, v in enumerate(self.item_ids)} if self.item_ids else {}
        self.item_emb = self._maybe(lambda: np.load(ART / "item_emb.npy"))
        self.faiss = None
        if self.item_emb is not None:
            import faiss
            faiss.omp_set_num_threads(1)   # avoid a torch/faiss OpenMP segfault on macOS arm64
            emb = self.item_emb.astype("float32"); faiss.normalize_L2(emb)
            self.faiss = faiss.IndexFlatIP(emb.shape[1]); self.faiss.add(emb)
        self.tower = self._maybe(self._load_tower)
        self.ranker = self._maybe(lambda: pickle.load(open(ART / "lgbm.pkl", "rb")))
        self.sim = self._maybe(lambda: json.load(open(ART / "similar_items.json")))
        fs = self._maybe(lambda: pickle.load(open(ART / "feature_store.pkl", "rb")))
        if fs:
            # Convert the store's dicts to pandas Series ONCE: featurize()
            # uses Series.map(...) per feature, and pandas rebuilds a hash
            # index from a raw dict on EVERY map call -- ~1.2s/request with
            # the full 551K-user store (vs ~1ms for the actual LightGBM
            # predict). Mapping against prebuilt Series is O(rows).
            import pandas as pd
            self.store = {k: (pd.Series(v) if isinstance(v, dict) else v)
                          for k, v in fs["store"].items()}
            self.now_ts = fs["now_ts"]
        else:
            self.store = None
            self.now_ts = None

    def _maybe(self, fn):
        try:
            return fn()
        except Exception as e:
            print("artifact not loaded:", e); return None

    def _load_tower(self):
        """TorchScripted user tower: (history_ids, mask) -> vector. No class needed."""
        import torch
        # map_location="cpu": the checkpoint may have been scripted on a CUDA runtime
        # (Colab) -- this container has no GPU, so loading without it fails outright.
        m = torch.jit.load(str(ART / "user_tower.pt"), map_location="cpu"); m.eval()
        def fn(hist_idx):
            if not hist_idx:
                return None
            import torch.nn.functional as F
            h = torch.tensor([hist_idx]); mask = torch.ones_like(h, dtype=torch.float32)
            with torch.no_grad():
                return F.normalize(m(h, mask), dim=-1)[0].numpy()
        return fn

    def popular(self, n=10, temperature=1.0, seed=None, mmr_lambda=1.0):
        """Weighted sample from the popularity head (not a fixed top-n) so
        the trending feed is different on every page load. `seed` comes from
        the frontend's per-browser-session feed seed: stable while the user
        browses, fresh on reload. mmr_lambda < 1 diversifies the pool with
        MMR first (relevance = popularity rank)."""
        pool = (self.popularity or [])[: max(n * 3, 60)]
        label = "Popular right now"
        if mmr_lambda < 0.999 and pool:
            rank_rel = {it: 1.0 - k / len(pool) for k, it in enumerate(pool)}
            pool = self.mmr(pool, rank_rel, n=len(pool), lam=mmr_lambda)
            label += f" + MMR(λ={mmr_lambda:g})"
        return _sample(pool, n, temperature, seed), label

    def recommend(self, user_id, history_item_ids, n=10, temperature=1.0, seed=None,
                  mmr_lambda=1.0):
        """Returns (item_ids, model_label, timings_ms). mmr_lambda < 1 applies
        Maximal Marginal Relevance diversification before sampling."""
        t = {}
        if not history_item_ids or self.faiss is None or self.tower is None:
            items, label = self.popular(n, temperature, seed)
            return items, label, t

        hist_idx = [self.idx_map[i] for i in history_item_ids if i in self.idx_map]

        s = time.perf_counter()
        uvec = self.tower(hist_idx)
        t["user_tower"] = (time.perf_counter() - s) * 1e3
        if uvec is None:
            items, label = self.popular(n, temperature, seed); return items, label, t

        s = time.perf_counter()
        import faiss
        q = uvec.astype("float32")[None, :]; faiss.normalize_L2(q)
        D, I = self.faiss.search(q, 100)
        cands = [self.item_ids[j] for j in I[0]]
        cand_scores = {self.item_ids[j]: float(D[0][k]) for k, j in enumerate(I[0])}
        t["faiss"] = (time.perf_counter() - s) * 1e3

        if self.ranker is None or self.store is None:        # retrieval-only (pre-S4)
            ordered, scores, label = cands, cand_scores, "Two-tower"
        else:
            s = time.perf_counter()
            ordered, scores = self._rerank(user_id, cands, cand_scores)
            t["lgbm"] = (time.perf_counter() - s) * 1e3
            label = "Two-tower + LightGBM"

        if mmr_lambda < 0.999:
            s = time.perf_counter()
            ordered = self.mmr(ordered, scores, n=len(ordered), lam=mmr_lambda)
            t["mmr"] = (time.perf_counter() - s) * 1e3
            label += f" + MMR(λ={mmr_lambda:g})"

        return _sample(ordered, n, temperature, seed), label, t

    def mmr(self, cands, rel_scores, n, lam=0.7):
        """Maximal Marginal Relevance re-ranking (bonus):
        pick items one by one, each maximizing
            lam * relevance(i)  -  (1 - lam) * max_similarity(i, already_picked)
        so the list stays relevant but avoids near-duplicates. Relevance is
        min-max-normalized to [0,1] so lam trades off against cosine
        similarity on a comparable scale. lam=1 -> pure relevance order."""
        kept = [c for c in cands if c in self.idx_map]
        if not kept or self.item_emb is None:
            return cands[:n]
        V = self.item_emb[[self.idx_map[c] for c in kept]].astype("float32")
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        rel = np.array([rel_scores.get(c, 0.0) for c in kept], dtype="float64")
        if rel.max() > rel.min():
            rel = (rel - rel.min()) / (rel.max() - rel.min())
        sims = V @ V.T
        selected, remaining = [], list(range(len(kept)))
        while remaining and len(selected) < n:
            if not selected:
                best = max(remaining, key=lambda i: rel[i])
            else:
                best = max(remaining,
                           key=lambda i: lam * rel[i] - (1 - lam) * sims[i, selected].max())
            selected.append(best)
            remaining.remove(best)
        return [kept[i] for i in selected]

    def _rerank(self, user_id, cands, cand_scores):
        """Compute ranking features online and re-order candidates.
        Returns (ordered_ids, {item_id: lgbm_score})."""
        import pandas as pd
        import sys; sys.path.insert(0, "/app")
        from src.ranking.features import featurize, FEATURE_NAMES
        df = pd.DataFrame({"user_id": [user_id] * len(cands),
                           "item_id": cands,
                           "retriever_score": [cand_scores.get(c, 0.0) for c in cands]})
        feats = featurize(df, self.store, self.now_ts or 0)
        scores = self.ranker.predict(feats[FEATURE_NAMES])
        order = np.argsort(-scores)
        return [cands[i] for i in order], {cands[i]: float(scores[i]) for i in order}

    def similar(self, item_id, n=10):
        if self.sim and item_id in self.sim:
            return self.sim[item_id][:n], "Similar items (embedding cosine)"
        return self.popular(n)


def _sample(items, n, temperature, seed=None, half_life=40.0):
    """Temperature sampling from the candidate list so recs vary on reload.

    `half_life` spreads the rank decay: the old e^-rank weighting collapsed
    to ~zero by rank 20, which made the "sampled" top-20 deterministic in
    practice. e^(-rank/8) keeps the head favored but the tail reachable.
    A fixed `seed` (per browser session) keeps the list stable while the
    user browses; a reload brings a new seed and a fresh list."""
    if temperature <= 0 or len(items) <= n:
        return items[:n]
    rng = np.random.default_rng(seed)
    ranks = np.arange(len(items))
    logits = -ranks / max(temperature * half_life, 1e-6)
    p = np.exp(logits - logits.max()); p /= p.sum()
    chosen = rng.choice(len(items), size=min(n, len(items)), replace=False, p=p)
    return [items[i] for i in sorted(chosen)]
