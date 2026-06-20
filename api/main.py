"""FastAPI service. Loads models + FAISS at startup, serves recommendations.
Swagger UI at /docs (required by the brief)."""
import time
from fastapi import FastAPI, HTTPException
from .db import get_conn, fetch_user_history, fetch_items
from .recommender import Recommender
from .schemas import RecResponse, Item

app = FastAPI(title="Electronics RecSys API", version="0.1.0")
rec = Recommender()                  # loads artifacts once at startup


def _to_items(conn, item_ids):
    meta = fetch_items(conn, item_ids)
    return [Item(**meta.get(i, {"item_id": i})) for i in item_ids]


@app.get("/health")
def health():
    return {"status": "ok", "artifacts": {k: getattr(rec, k) is not None
            for k in ["popularity", "faiss", "tower", "ranker", "sim"]}}


@app.get("/popular", response_model=RecResponse)
def popular(n: int = 10):
    t0 = time.perf_counter()
    ids, label = rec.popular(n)
    conn = get_conn()
    try:
        items = _to_items(conn, ids)
    finally:
        conn.close()
    return RecResponse(model_label=label, items=items,
                       latency_ms={"total": (time.perf_counter() - t0) * 1e3})


@app.get("/recommend/{user_id}", response_model=RecResponse)
def recommend(user_id: str, n: int = 10, temperature: float = 1.0):
    t0 = time.perf_counter()
    conn = get_conn()
    try:
        s = time.perf_counter()
        history = fetch_user_history(conn, user_id, limit=20)
        db_ms = (time.perf_counter() - s) * 1e3
        ids, label, timings = rec.recommend(user_id, history, n=n, temperature=temperature)
        items = _to_items(conn, ids)
    finally:
        conn.close()
    timings["db"] = db_ms
    timings["total"] = (time.perf_counter() - t0) * 1e3
    return RecResponse(model_label=label, items=items, latency_ms=timings)


@app.get("/similar/{item_id}", response_model=RecResponse)
def similar(item_id: str, n: int = 10):
    t0 = time.perf_counter()
    ids, label = rec.similar(item_id, n)
    conn = get_conn()
    try:
        items = _to_items(conn, ids)
    finally:
        conn.close()
    return RecResponse(model_label=label, items=items,
                       latency_ms={"total": (time.perf_counter() - t0) * 1e3})


@app.get("/because-you-liked/{user_id}", response_model=RecResponse)
def because_you_liked(user_id: str, n: int = 10):
    """Pick one item from the user's history, show items similar to THAT item."""
    t0 = time.perf_counter()
    conn = get_conn()
    try:
        history = fetch_user_history(conn, user_id, limit=20)
        if not history:
            ids, label = rec.popular(n)
            seed = None
        else:
            seed = history[0]
            ids, label = rec.similar(seed, n)
            label = f"Because you liked this"
        items = _to_items(conn, ids)
    finally:
        conn.close()
    return RecResponse(model_label=label, items=items,
                       latency_ms={"total": (time.perf_counter() - t0) * 1e3, "seed_item": seed})
