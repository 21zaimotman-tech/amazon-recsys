"""Feature engineering for the ranker. The brief wants AT LEAST 10 distinct
features across user / item / cross / temporal. Below are 12 — document each
one's rationale in the notebook.

Training rows = (user, candidate_item) pairs from the best retriever's top-100
for train users, labelled 1 if the user actually interacted (in ground truth)
else 0. Compute all features from TRAINING data only (no test leakage)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_feature_store(train_df, item_meta):
    """Precompute the per-user and per-item aggregates features read from."""
    store = {}
    # ---- item-level ----
    store["item_pop"] = train_df["item_id"].value_counts().to_dict()           # 1 popularity
    store["item_avg_rating"] = train_df.groupby("item_id")["rating"].mean().to_dict()  # 2
    last_ts = train_df.groupby("item_id")["timestamp"].max().to_dict()
    store["item_recency"] = last_ts                                            # 3 item freshness
    meta = item_meta.set_index("item_id")
    store["item_price"] = meta["price"].to_dict()                              # 4 price
    store["item_brand"] = meta["brand"].to_dict()
    store["item_cat"] = meta["category"].to_dict()
    # ---- user-level ----
    store["user_activity"] = train_df["user_id"].value_counts().to_dict()      # 5 activity count
    store["user_avg_rating"] = train_df.groupby("user_id")["rating"].mean().to_dict()  # 6
    store["user_last_ts"] = train_df.groupby("user_id")["timestamp"].max().to_dict()
    # user preferred category / brand / typical price (for cross-features)
    j = train_df.merge(meta[["category", "brand", "price"]], left_on="item_id",
                       right_index=True, how="left")
    store["user_top_cat"] = j.groupby("user_id")["category"].agg(
        lambda s: s.value_counts().index[0] if len(s.dropna()) else None).to_dict()
    store["user_top_brand"] = j.groupby("user_id")["brand"].agg(
        lambda s: s.value_counts().index[0] if len(s.dropna()) else None).to_dict()
    store["user_avg_price"] = j.groupby("user_id")["price"].mean().to_dict()
    store["global_avg_rating"] = float(train_df["rating"].mean())
    return store


def featurize(pairs: pd.DataFrame, store: dict, now_ts: int) -> pd.DataFrame:
    """pairs has columns [user_id, item_id, retriever_score]. Returns a feature
    matrix. 12 features spanning the four required groups."""
    g = store
    f = pd.DataFrame(index=pairs.index)
    u, it = pairs["user_id"], pairs["item_id"]
    f["retriever_score"] = pairs["retriever_score"]                                   # signal from retrieval
    f["item_pop"] = it.map(g["item_pop"]).fillna(0)                                   # item
    f["item_avg_rating"] = it.map(g["item_avg_rating"]).fillna(g["global_avg_rating"])# item
    f["item_price"] = it.map(g["item_price"])                                         # item
    f["item_age_days"] = (now_ts - it.map(g["item_recency"]).fillna(now_ts)) / 8.64e7 # temporal
    f["user_activity"] = u.map(g["user_activity"]).fillna(0)                          # user
    f["user_avg_rating"] = u.map(g["user_avg_rating"]).fillna(g["global_avg_rating"]) # user
    f["days_since_user_last"] = (now_ts - u.map(g["user_last_ts"]).fillna(now_ts)) / 8.64e7  # temporal
    # ---- cross-features ----
    f["cat_match"] = (u.map(g["user_top_cat"]) == it.map(g["item_cat"])).astype(int)  # cross
    f["brand_match"] = (u.map(g["user_top_brand"]) == it.map(g["item_brand"])).astype(int)  # cross
    up = u.map(g["user_avg_price"]); ip = it.map(g["item_price"])
    f["price_gap"] = (ip - up).abs()                                                  # cross
    f["rating_vs_user_mean"] = it.map(g["item_avg_rating"]).fillna(g["global_avg_rating"]) \
                               - u.map(g["user_avg_rating"]).fillna(g["global_avg_rating"])  # cross
    return f.fillna(0.0)


FEATURE_NAMES = ["retriever_score", "item_pop", "item_avg_rating", "item_price",
                 "item_age_days", "user_activity", "user_avg_rating",
                 "days_since_user_last", "cat_match", "brand_match",
                 "price_gap", "rating_vs_user_mean"]
