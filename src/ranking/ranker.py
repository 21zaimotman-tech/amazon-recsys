"""LightGBM ranker with the LambdaRank objective, trained on retriever
candidates. Compare NDCG@10 and Coverage against retrieval-only ranking (the
ablation in ANALYSIS.md)."""
from __future__ import annotations
import numpy as np
import lightgbm as lgb
from .features import FEATURE_NAMES


def train_ranker(X_train, y_train, group_train, X_val=None, y_val=None, group_val=None):
    """group_* = number of candidate rows per user (LambdaRank needs query groups).
    X are feature matrices, y are 0/1 relevance labels."""
    train_set = lgb.Dataset(X_train, label=y_train, group=group_train)
    valid_sets = [train_set]
    if X_val is not None:
        valid_sets.append(lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_set))
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10],
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        verbose=-1,
        # macOS: LightGBM's OpenMP pool segfaults when torch+faiss have
        # already loaded their own OpenMP runtimes into the process (same
        # conflict faiss_index.py guards with omp_set_num_threads(1)).
        # Single-threaded training is plenty fast at this data size.
        num_threads=1,
    )
    model = lgb.train(params, train_set, num_boost_round=500, valid_sets=valid_sets,
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    return model


def rerank(model, feature_df, item_ids, top_k=10):
    """Score candidates and return the top_k item_ids best-first."""
    scores = model.predict(feature_df[FEATURE_NAMES], num_threads=1)  # same OpenMP guard as training
    order = np.argsort(-scores)
    return [item_ids[i] for i in order[:top_k]], scores[order[:top_k]]


def feature_importance(model):
    imp = model.feature_importance(importance_type="gain")
    return sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])
