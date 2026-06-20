"""Baselines on the frozen split -> Recall@20/50, NDCG@10, Coverage.
Run after build_dataset.py. The same calls go into notebook 02 with narration.

  python scripts/run_baselines.py
"""
import sys; sys.path.insert(0, ".")
import pandas as pd
from src import config as C
from src.baselines.popularity import PopularityRecommender, RandomRecommender
from src.data.split import ground_truth_from
from src.eval.metrics import evaluate


def main():
    train = pd.read_parquet("data/train.parquet")
    test = pd.read_parquet("data/test.parquet")

    catalog = train.item_id.nunique()                  # recommendable universe = train items
    test_gt = ground_truth_from(test, positive_only=True)   # test positives (rating >= 4)
    users = list(test_gt.keys())
    seen = train.groupby("user_id")["item_id"].agg(set).to_dict()  # exclude train-seen
    N = max(C.K_VALUES)

    print(f"catalog={catalog:,}  test users scored={len(users):,}\n")
    rows = []
    for name, Model in [("Random", RandomRecommender), ("Popularity", PopularityRecommender)]:
        model = Model().fit(train)
        recs = model.recommend(users, n=N, exclude=seen)
        m = evaluate(recs, test_gt, catalog_size=catalog, k_values=C.K_VALUES)
        m["method"] = name
        rows.append(m)

    df = pd.DataFrame(rows).set_index("method")
    cols = [c for c in df.columns if c != "n_users_scored"]
    print(df[cols].round(4).to_string())


if __name__ == "__main__":
    main()
