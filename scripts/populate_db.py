"""Populate Postgres from the frozen split + item metadata.
items     <- item metadata (with image_url for the webapp)
interactions <- TEST-PERIOD interactions (used as serving-time user history)
Run once after `docker compose up postgres`."""
import os, psycopg2
import pandas as pd
from psycopg2.extras import execute_values

def populate(items_parquet, test_parquet):
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "recsys"),
        user=os.getenv("POSTGRES_USER", "recsys"),
        password=os.getenv("POSTGRES_PASSWORD", "recsys"))
    items = pd.read_parquet(items_parquet)
    test = pd.read_parquet(test_parquet)
    with conn.cursor() as cur:
        item_cols = items[["item_id","title","image_url","category","brand","price","avg_rating"]]
        # .where(notnull, None) on a float64 column silently coerces None back to NaN (the
        # column dtype can't hold None) -> NaN would land in Postgres as a literal NaN instead
        # of NULL, and the API's JSON encoder rejects NaN at serialization time. astype(object)
        # first so None actually sticks.
        item_rows = item_cols.astype(object).where(pd.notnull(item_cols), None).values.tolist()
        execute_values(cur,
            "INSERT INTO items (item_id,title,image_url,category,brand,price,avg_rating)"
            " VALUES %s ON CONFLICT (item_id) DO NOTHING",
            item_rows)
        execute_values(cur,
            "INSERT INTO interactions (user_id,item_id,rating,ts) VALUES %s",
            test[["user_id","item_id","rating","timestamp"]].values.tolist())
    conn.commit(); conn.close()
    print("DB populated:", len(items), "items,", len(test), "interactions")

if __name__ == "__main__":
    populate("data/items.parquet", "data/test.parquet")
