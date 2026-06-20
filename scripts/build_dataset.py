"""Build the frozen dataset: stream -> subsample -> time-split -> save parquet.

Reads the raw JSONL files directly from HuggingFace (no `datasets` library, so
it's immune to the "Dataset scripts are no longer supported" breakage in
datasets>=4.0). You can also point it at locally-downloaded files.

  # fast dry-run (streams only the first --limit rows, ~minutes):
  python scripts/build_dataset.py --limit 200000 --meta-scan-cap 1000000

  # full run from already-downloaded files (recommended for the real pass):
  python scripts/build_dataset.py --reviews-file Electronics.jsonl --meta-file meta_Electronics.jsonl

Outputs into ./data: train.parquet, val.parquet, test.parquet, items.parquet
"""
import argparse, json
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
import sys; sys.path.insert(0, ".")
from src import config as C
from src.data.load import subsample, _clean_price
from src.data.split import time_split, warn_leakage

BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
REVIEW_URL = f"{BASE}/review_categories/{C.CATEGORY}.jsonl"
META_URL = f"{BASE}/meta_categories/meta_{C.CATEGORY}.jsonl"


def stream_jsonl(url=None, local_file=None):
    """Yield dicts from a JSONL source: a local file if given, else stream URL."""
    if local_file:
        with open(local_file, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    yield json.loads(line)
    else:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if raw:
                    yield json.loads(raw)


def stream_reviews(limit=None, local_file=None):
    cutoff = int(datetime(C.RECENT_FROM_YEAR, 1, 1).timestamp() * 1000)  # ms
    rows = []
    for r in stream_jsonl(REVIEW_URL, local_file):
        if r["timestamp"] >= cutoff:
            rows.append((r["user_id"], r[C.USE_ITEM_LEVEL], r["rating"], r["timestamp"]))
        if limit and len(rows) >= limit:
            break
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def pick_image(imgs):
    """Return one image URL. Raw JSONL stores `images` as a LIST of per-image
    dicts ({'thumb','large','hi_res','variant'}); the script-built version used
    a DICT of lists. Handle both, skipping None entries."""
    if isinstance(imgs, dict):
        for k in ("large", "hi_res", "thumb"):
            for u in (imgs.get(k) or []):
                if u:
                    return u
    elif isinstance(imgs, list):
        for entry in imgs:
            if isinstance(entry, dict):
                for k in ("large", "hi_res", "thumb"):
                    if entry.get(k):
                        return entry[k]
    return None


def stream_meta(needed_ids, scan_cap=None, local_file=None):
    """Keep only items we use; stop once all are found (or scan_cap hit)."""
    needed = set(needed_ids)
    rows, seen = [], 0
    for m in stream_jsonl(META_URL, local_file):
        seen += 1
        pid = m.get("parent_asin")
        if pid in needed:
            cats = m.get("categories") or []
            rows.append({
                "item_id": pid,
                "title": m.get("title"),
                "image_url": pick_image(m.get("images")),
                "category": cats[-1] if cats else m.get("main_category"),
                "brand": m.get("store"),
                "price": _clean_price(m.get("price")),
                "avg_rating": m.get("average_rating"),
            })
            needed.discard(pid)
        if not needed or (scan_cap and seen >= scan_cap):
            break
    return pd.DataFrame(rows).drop_duplicates("item_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap raw reviews kept (dry-run)")
    ap.add_argument("--meta-scan-cap", type=int, default=None, help="cap metadata rows scanned")
    ap.add_argument("--reviews-file", default=None, help="local Electronics.jsonl (skip URL stream)")
    ap.add_argument("--meta-file", default=None, help="local meta_Electronics.jsonl")
    args = ap.parse_args()
    data = Path("data"); data.mkdir(exist_ok=True)

    print(f"streaming reviews from {'file' if args.reviews_file else 'HuggingFace'} ...")
    df = stream_reviews(args.limit, args.reviews_file)
    print(f"  raw kept: {len(df):,}")
    if df.empty:
        sys.exit("No rows kept. If the file isn't empty, check the timestamp unit "
                 "(this assumes 13-digit ms) or lower RECENT_FROM_YEAR in src/config.py.")

    df = subsample(df)
    print(f"  after subsample: {len(df):,} interactions | "
          f"users={df.user_id.nunique():,} items={df.item_id.nunique():,}")

    train, val, test = time_split(df)
    warn_leakage(train, test)
    print(f"  split -> train={len(train):,}  val={len(val):,}  test={len(test):,}")

    print("streaming item metadata (stops once all kept items are found) ...")
    meta = stream_meta(df.item_id.unique(), args.meta_scan_cap, args.meta_file)
    print(f"  items with metadata: {len(meta):,} / {df.item_id.nunique():,}")

    for name, d in [("train", train), ("val", val), ("test", test), ("items", meta)]:
        d.to_parquet(data / f"{name}.parquet")
    print("saved -> data/{train,val,test,items}.parquet")


if __name__ == "__main__":
    main()
