"""Clean raw fields and subsample interactions before the time split.

Called from scripts/build_dataset.py right after the recent-window filter:
_clean_price by stream_meta (per-row), subsample on the whole interactions df."""
from __future__ import annotations
import re
import pandas as pd
from src import config as C


def _clean_price(price):
    """Raw metadata price is a messy field: None, a plain number, "$49.99",
    or a range like "$10.00 - $20.00". Return a float (midpoint for ranges)
    or None if nothing numeric is present."""
    if price is None:
        return None
    if isinstance(price, (int, float)):
        return float(price) if price == price else None  # NaN != NaN
    s = str(price).strip()
    if not s or s.lower() in ("none", "nan", ""):
        return None
    nums = re.findall(r"[\d,]+\.?\d*", s)
    if not nums:
        return None
    try:
        vals = [float(n.replace(",", "")) for n in nums if n.replace(",", "").replace(".", "")]
    except ValueError:
        return None
    return sum(vals) / len(vals) if vals else None


def subsample(df: pd.DataFrame) -> pd.DataFrame:
    """Iterative k-core filtering, then the mandatory 5M hard cap.

    The recent-window filter is already applied by the caller. Here:
    repeatedly drop users below C.KCORE_USER and items below C.KCORE_ITEM
    until the frame stops shrinking (dropping items can push users below
    the threshold and vice versa, so a single pass isn't enough), then
    truncate to the most recent C.MAX_INTERACTIONS rows if still over.
    """
    print(f"    subsample start: {len(df):,} interactions | "
          f"users={df.user_id.nunique():,} items={df.item_id.nunique():,}")

    prev_len = None
    round_n = 0
    while prev_len != len(df):
        prev_len = len(df)
        round_n += 1
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= C.KCORE_ITEM].index)]
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= C.KCORE_USER].index)]
    print(f"    after {C.KCORE_USER}-core ({round_n} rounds): {len(df):,} interactions | "
          f"users={df.user_id.nunique():,} items={df.item_id.nunique():,}")

    if len(df) > C.MAX_INTERACTIONS:
        df = df.sort_values("timestamp").tail(C.MAX_INTERACTIONS)
        print(f"    after {C.MAX_INTERACTIONS:,} cap (most recent kept): {len(df):,} interactions | "
              f"users={df.user_id.nunique():,} items={df.item_id.nunique():,}")

    return df.reset_index(drop=True)


def encode_ids(train_df, val_df, test_df):
    """Fit user/item id -> contiguous-int encoders on TRAIN ids only, then add
    `u`/`i` integer columns to all three splits. `MFBPR`/`TwoTower` embedding
    tables index by these, never by the raw string ids.

    A val/test id that never appears in train (a cold user/item) encodes to
    a trailing <UNK> index (`n_users` / `n_items`, i.e. one past the last
    valid row) instead of NaN — callers must filter these out before scoring
    (the API's `idx_map` lookup does the same: unknown id -> skip -> fall
    back to popularity). Embedding tables should therefore be sized
    `n_users + 1` / `n_items + 1` to keep that trailing UNK row addressable.

    Returns (train, val, test, user_ids, item_ids) where user_ids/item_ids
    are lists with `list[encoded_int] == original_id` — item_ids is exactly
    the ordering `scripts/export_artifacts.py` writes to `item_encoder.json`.
    """
    user_ids = sorted(train_df["user_id"].unique().tolist())
    item_ids = sorted(train_df["item_id"].unique().tolist())
    user2idx = {u: n for n, u in enumerate(user_ids)}
    item2idx = {it: n for n, it in enumerate(item_ids)}
    n_users, n_items = len(user_ids), len(item_ids)

    def _apply(df):
        df = df.copy()
        df["u"] = df["user_id"].map(user2idx).fillna(n_users).astype(int)
        df["i"] = df["item_id"].map(item2idx).fillna(n_items).astype(int)
        return df

    return _apply(train_df), _apply(val_df), _apply(test_df), user_ids, item_ids
