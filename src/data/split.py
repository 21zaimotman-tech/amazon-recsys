"""Mandatory time-based split (a random split scores 0 on "Data prep &
baselines" per the brief) and ground-truth construction shared by every
notebook's evaluation."""
from __future__ import annotations
import pandas as pd
from src import config as C


def time_split(df: pd.DataFrame):
    """Sort by timestamp, cut at GLOBAL quantiles: train < VAL_QUANTILE <=
    val < TEST_QUANTILE <= test. Quantile cuts (not row counts) so the split
    reflects the actual timeline regardless of how bursty activity is."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    val_cut = df["timestamp"].quantile(C.VAL_QUANTILE)
    test_cut = df["timestamp"].quantile(C.TEST_QUANTILE)
    train = df[df["timestamp"] < val_cut].reset_index(drop=True)
    val = df[(df["timestamp"] >= val_cut) & (df["timestamp"] < test_cut)].reset_index(drop=True)
    test = df[df["timestamp"] >= test_cut].reset_index(drop=True)
    return train, val, test


def warn_leakage(train: pd.DataFrame, test: pd.DataFrame):
    """Sanity check, not a hard failure: train must never contain timestamps
    later than the earliest test timestamp. Prints instead of raising so a
    tiny --limit dry run doesn't die on a degenerate split."""
    if train.empty or test.empty:
        return
    train_max, test_min = train["timestamp"].max(), test["timestamp"].min()
    if train_max > test_min:
        print(f"WARNING: possible leakage — train.timestamp.max()={train_max} "
              f"> test.timestamp.min()={test_min}")


def ground_truth_from(df: pd.DataFrame, positive_only: bool = True) -> dict:
    """{user_id -> set(item_id)}. When positive_only, keeps only rows with
    rating >= C.POSITIVE_RATING_THRESHOLD — the "liked" definition used by
    every baseline/retriever/ranker evaluation in this project."""
    d = df[df["rating"] >= C.POSITIVE_RATING_THRESHOLD] if positive_only else df
    return d.groupby("user_id")["item_id"].agg(set).to_dict()
