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
        self.store = fs["store"] if fs else None
        self.now_ts = fs["now_ts"] if fs else None

    def _maybe(self, fn):
        try:
            return fn()
        except Exception as e:
            print("artifact not loaded:", e); return None

    def _load_tower(self):
        """TorchScripted user tower: (history_ids, mask) -> vector. No class needed."""
        import torch
        m = torch.jit.load(str(ART / "user_tower.pt")); m.eval()
        def fn(hist_idx):
            if not hist_idx:
                return None
            import torch.nn.functional as F
            h = torch.tensor([hist_idx]); mask = torch.ones_like(h, dtype=torch.float32)
            with torch.no_grad():
                return F.normalize(m(h, mask), dim=-1)[0].numpy()
        return fn

    def popular(self, n=10):
        return (self.popularity or [])[:n], "Popular right now"

    def recommend(self, user_id, history_item_ids, n=10, temperature=1.0):
        """Returns (item_ids, model_label, timings_ms)."""
        t = {}
        if not history_item_ids or self.faiss is None or self.tower is None:
            items, label = self.popular(n)
            return items, label, t

        hist_idx = [self.idx_map[i] for i in history_item_ids if i in self.idx_map]

        s = time.perf_counter()
        uvec = self.tower(hist_idx)
        t["user_tower"] = (time.perf_counter() - s) * 1e3
        if uvec is None:
            items, label = self.popular(n); return items, label, t

        s = time.perf_counter()
        import faiss
        q = uvec.astype("float32")[None, :]; faiss.normalize_L2(q)
        D, I = self.faiss.search(q, 100)
        cands = [self.item_ids[j] for j in I[0]]
        cand_scores = {self.item_ids[j]: float(D[0][k]) for k, j in enumerate(I[0])}
        t["faiss"] = (time.perf_counter() - s) * 1e3

        if self.ranker is None or self.store is None:        # retrieval-only (pre-S4)
            return _sample(cands, n, temperature), "Two-tower", t

        s = time.perf_counter()
        ranked = self._rerank(user_id, cands, cand_scores)
        t["lgbm"] = (time.perf_counter() - s) * 1e3
        return _sample(ranked, n, temperature), "Two-tower + LightGBM", t

    def _rerank(self, user_id, cands, cand_scores):
        """Compute ranking features online and re-order candidates."""
        import pandas as pd
        import sys; sys.path.insert(0, "/app")
        from src.ranking.features import featurize, FEATURE_NAMES
        df = pd.DataFrame({"user_id": [user_id] * len(cands),
                           "item_id": cands,
                           "retriever_score": [cand_scores.get(c, 0.0) for c in cands]})
        feats = featurize(df, self.store, self.now_ts or 0)
        scores = self.ranker.predict(feats[FEATURE_NAMES])
        order = np.argsort(-scores)
        return [cands[i] for i in order]

    def similar(self, item_id, n=10):
        if self.sim and item_id in self.sim:
            return self.sim[item_id][:n], "Similar items (embedding cosine)"
        return self.popular(n)


def _sample(items, n, temperature):
    """Temperature sampling from the candidate list so recs vary on reload."""
    if temperature <= 0 or len(items) <= n:
        return items[:n]
    ranks = np.arange(len(items))
    logits = -ranks / max(temperature, 1e-6)
    p = np.exp(logits - logits.max()); p /= p.sum()
    chosen = np.random.choice(len(items), size=min(n, len(items)), replace=False, p=p)
    return [items[i] for i in sorted(chosen)]
