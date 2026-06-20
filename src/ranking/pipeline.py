"""Build LightGBM training data from retriever candidates.

LEAKAGE-SAFE LABELING (important design choice — explain at defense):
  - The retriever is trained on TRAIN.
  - For ranker training we retrieve candidates for TRAIN users and label a
    candidate POSITIVE if the user interacted with it in the VALIDATION period
    (held-out, strictly *after* train), else NEGATIVE.
  - Features are computed from TRAIN only.
  This makes the ranker learn to predict *future* engagement from *past* signal,
  exactly like deployment — instead of trivially memorising train positives.
Evaluate the full retrieve->rank pipeline on the TEST period.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import featurize, FEATURE_NAMES


def build_ranker_data(candidate_recs, label_gt, store, now_ts, retriever_scores=None):
    """candidate_recs : {train_user -> [item_id,...]}  (train-seen already excluded)
       label_gt       : {user -> set(item_id)} positives from the VALIDATION period
       store          : feature store from features.build_feature_store
       retriever_scores (optional): {(user,item) -> float}; defaults to -rank

    Returns X[FEATURE_NAMES], y (0/1), group (rows-per-user list), and the raw
    pairs DataFrame. Groups with no positive among candidates are dropped
    (LambdaRank needs at least one positive per query)."""
    rows, group = [], []
    for u, items in candidate_recs.items():
        pos = label_gt.get(u, set())
        if not pos.intersection(items):
            continue
        for rank, it in enumerate(items):
            s = (retriever_scores or {}).get((u, it), -rank)
            rows.append((u, it, s, 1 if it in pos else 0))
        group.append(len(items))
    pairs = pd.DataFrame(rows, columns=["user_id", "item_id", "retriever_score", "label"])
    X = featurize(pairs[["user_id", "item_id", "retriever_score"]], store, now_ts)
    return X[FEATURE_NAMES], pairs["label"].values, group, pairs
