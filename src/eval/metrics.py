"""Shared ranking metrics. IMPLEMENT ONCE, IMPORT EVERYWHERE.

Every model (baselines, MF-BPR, two-tower, LightGBM) is scored through these
exact functions so the comparison tables in the notebook and ANALYSIS.md are
consistent.

Conventions
-----------
recommendations : dict[user_id -> list[item_id]]   ranked best-first, length >= max(k)
ground_truth    : dict[user_id -> set[item_id]]     the held-out positives for that user
catalog_size    : int                               number of distinct items in the catalog

Users with no ground-truth positives are skipped (they cannot be scored).
All metrics are macro-averaged over scored users.
"""
from __future__ import annotations
import numpy as np


def _dcg(hits: np.ndarray) -> float:
    """Discounted Cumulative Gain for a binary hit vector (hits[i] in {0,1}).
    gain at rank i (0-indexed) is hits[i] / log2(i + 2)."""
    discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    return float(np.sum(hits * discounts))


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """NDCG@k for one user. IDCG is the DCG of the best possible ordering,
    i.e. min(len(relevant), k) hits stacked at the top."""
    if not relevant:
        return 0.0
    topk = recommended[:k]
    hits = np.array([1.0 if item in relevant else 0.0 for item in topk])
    dcg = _dcg(hits)
    ideal_hits = np.ones(min(len(relevant), k))
    idcg = _dcg(ideal_hits)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of a user's relevant items captured in the top-k.
    Denominator is |relevant| (capped at k is an alternative convention - we
    use |relevant| and note it in ANALYSIS.md)."""
    if not relevant:
        return 0.0
    topk = set(recommended[:k])
    return len(topk & relevant) / len(relevant)


def catalog_coverage(recommendations: dict, catalog_size: int, k: int) -> float:
    """% of the catalog that appears at least once across all users' top-k."""
    recommended_items = set()
    for items in recommendations.values():
        recommended_items.update(items[:k])
    return len(recommended_items) / catalog_size if catalog_size else 0.0


def evaluate(recommendations: dict,
             ground_truth: dict,
             catalog_size: int,
             k_values=(10, 20, 50)) -> dict:
    """Run the full metric suite. Returns a flat dict ready for a DataFrame row.

    Reports Recall@k and NDCG@k for every k in k_values, plus Coverage@max(k).
    Matches the brief: Recall@20, Recall@50, NDCG@10, Catalog Coverage.
    """
    scored_users = [u for u in recommendations if ground_truth.get(u)]
    if not scored_users:
        raise ValueError("No users with ground-truth positives to score.")

    out = {}
    for k in k_values:
        recalls = [recall_at_k(recommendations[u], ground_truth[u], k) for u in scored_users]
        ndcgs = [ndcg_at_k(recommendations[u], ground_truth[u], k) for u in scored_users]
        out[f"Recall@{k}"] = float(np.mean(recalls))
        out[f"NDCG@{k}"] = float(np.mean(ndcgs))

    out[f"Coverage@{max(k_values)}"] = catalog_coverage(
        recommendations, catalog_size, max(k_values))
    out["n_users_scored"] = len(scored_users)
    return out
