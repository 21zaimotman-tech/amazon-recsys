"""Batch evaluation harness — the glue between a trained model and metrics.py.

A retriever gives you item embeddings + a way to produce a user vector. This
module turns that into a {user -> ranked items} dict (excluding items the user
already saw in training) and scores it. Used by notebooks 02-05.
"""
from __future__ import annotations
import numpy as np
from .metrics import evaluate


def seen_items(train_df) -> dict:
    """{user_id -> set(item_id)} seen during training. Excluded from recs so we
    don't get credit for re-recommending known items (and metrics stay honest)."""
    return train_df.groupby("user_id")["item_id"].agg(set).to_dict()


def recommend_from_index(index, user_vecs, user_ids, seen=None, n=50, retrieve=200):
    """index: EmbeddingIndex. user_vecs: (U, dim) aligned with user_ids.
    Retrieve `retrieve` candidates, drop training-seen items, keep top-n."""
    seen = seen or {}
    raw = index.search(np.asarray(user_vecs, dtype="float32"), retrieve)
    recs = {}
    for uid, items in zip(user_ids, raw):
        s = seen.get(uid, ())
        recs[uid] = [it for it in items if it not in s][:n]
    return recs


def evaluate_index(index, user_vecs, user_ids, test_gt, train_df, catalog_size,
                   k_values=(10, 20, 50), retrieve=200):
    """One call: recommend for all test users, then score. Returns (recs, metrics)."""
    recs = recommend_from_index(index, user_vecs, user_ids, seen_items(train_df),
                                n=max(k_values), retrieve=retrieve)
    return recs, evaluate(recs, test_gt, catalog_size, k_values)


# ---- user-vector builders (call from the notebook) ----
def mfbpr_user_vecs(model, user_idx):
    """MF-BPR user vectors = rows of the user embedding table, with a constant
    1 appended so the dot product with export_item_embeddings' augmented item
    vectors reproduces dot(user,item) + item_bias exactly."""
    vecs = model.user_emb.weight.detach().cpu().numpy()[user_idx]
    ones = np.ones((vecs.shape[0], 1), dtype=vecs.dtype)
    return np.concatenate([vecs, ones], axis=1)


def two_tower_user_vecs(model, histories, max_hist=20, device="cpu"):
    """histories: list[list[int encoded item ids]] (each user's TRAIN history,
    or TEST-period history when simulating serving). Returns (U, dim) normalised."""
    import torch
    import torch.nn.functional as F
    vecs = []
    model.eval()
    with torch.no_grad():
        for h in histories:
            h = h[-max_hist:]
            if not h:
                vecs.append(np.zeros(model.item_tower.emb.embedding_dim, dtype="float32"))
                continue
            t = torch.tensor([h], device=device)
            m = torch.ones_like(t, dtype=torch.float32)
            v = F.normalize(model.user_tower(t, m), dim=-1)[0].cpu().numpy()
            vecs.append(v)
    return np.vstack(vecs)
