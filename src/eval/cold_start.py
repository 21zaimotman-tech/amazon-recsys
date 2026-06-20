"""Cold-start analysis (ANALYSIS.md Section 6): performance by user activity.
Bucket test users by how many TRAIN interactions they had, score each bucket."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import recall_at_k, ndcg_at_k

DEFAULT_EDGES = ((1, 4), (5, 19), (20, np.inf))


def _labeler(train_df, edges):
    counts = train_df["user_id"].value_counts().to_dict()
    def label(u):
        c = counts.get(u, 0)
        for lo, hi in edges:
            if lo <= c <= hi:
                return f"{lo}-{'+' if hi == np.inf else int(hi)}"
        return "0"
    return label


def evaluate_by_bucket(recs, test_gt, train_df, k=20, edges=DEFAULT_EDGES):
    """Returns a DataFrame: bucket, users, Recall@k, NDCG@k. Run once per model
    (e.g. popularity vs two-tower) and put them side by side."""
    label = _labeler(train_df, edges)
    rows = {}
    for u, rel in test_gt.items():
        if u not in recs or not rel:
            continue
        rows.setdefault(label(u), []).append(
            (recall_at_k(recs[u], rel, k), ndcg_at_k(recs[u], rel, k)))
    out = []
    for b, vals in rows.items():
        a = np.array(vals)
        out.append({"bucket": b, "users": len(vals),
                    f"Recall@{k}": a[:, 0].mean(), f"NDCG@{k}": a[:, 1].mean()})
    return pd.DataFrame(out).sort_values("bucket").reset_index(drop=True)
