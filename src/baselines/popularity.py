"""Popularity & Random baselines. Popularity is a STRONG baseline on Electronics
because reviews concentrate on head items — beating it on Recall is hard, beating
it on Coverage is easy. That contrast is a real finding for ANALYSIS.md."""
from __future__ import annotations
import numpy as np
from collections import Counter


class PopularityRecommender:
    def __init__(self):
        self.ranked_items = []

    def fit(self, train_df):
        counts = Counter(train_df["item_id"])
        self.ranked_items = [it for it, _ in counts.most_common()]
        return self

    def recommend(self, users, n=100, exclude=None):
        """Same popular list for everyone, minus items the user already saw."""
        exclude = exclude or {}
        recs = {}
        for u in users:
            seen = exclude.get(u, set())
            recs[u] = [it for it in self.ranked_items if it not in seen][:n]
        return recs


class RandomRecommender:
    def __init__(self, seed=42):
        self.items = []
        self.rng = np.random.default_rng(seed)

    def fit(self, train_df):
        self.items = train_df["item_id"].unique().tolist()
        return self

    def recommend(self, users, n=100, exclude=None):
        exclude = exclude or {}
        recs = {}
        for u in users:
            seen = exclude.get(u, set())
            pool = [it for it in self.items if it not in seen]
            self.rng.shuffle(pool)
            recs[u] = pool[:n]
        return recs
