"""FastAPI service. Loads models + FAISS at startup, serves recommendations.
Swagger UI at /docs (required by the brief)."""
import time
import psycopg2.errors
from fastapi import FastAPI, HTTPException
from .db import (get_conn, fetch_user_history, fetch_items, search_items, create_user,
                 get_user, log_event, add_to_cart, remove_from_cart, get_cart, checkout,
                 add_to_wishlist, remove_from_wishlist, get_wishlist, get_orders,
                 create_session, get_session_user, delete_session)
from .auth import hash_password, verify_password
from .recommender import Recommender
from .schemas import (RecResponse, Item, RegisterRequest, LoginRequest, AuthResponse,
                      CartResponse, OrderItem, OrdersResponse)

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


@app.get("/search", response_model=RecResponse)
def search(q: str, n: int = 20):
    """Full-catalog search (title/brand/category) -- distinct from the
    frontend's per-page filter, which only ever searches whatever's already
    loaded. Not a recommendation, so model_label says so plainly."""
    t0 = time.perf_counter()
    conn = get_conn()
    try:
        rows = search_items(conn, q, limit=n)
    finally:
        conn.close()
    return RecResponse(model_label=f'Search results for "{q}"', items=[Item(**r) for r in rows],
                       latency_ms={"total": (time.perf_counter() - t0) * 1e3})


# ---------------------------------------------------------------- accounts
@app.post("/auth/register", response_model=AuthResponse, status_code=201)
def register(body: RegisterRequest):
    if not body.user_id.strip() or not body.password:
        raise HTTPException(400, "user_id and password are required")
    password_hash, password_salt = hash_password(body.password)
    conn = get_conn()
    try:
        create_user(conn, body.user_id, password_hash, password_salt)
        token = create_session(conn, body.user_id)
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(409, f"user_id '{body.user_id}' is already taken")
    finally:
        conn.close()
    return AuthResponse(user_id=body.user_id, n_interactions=0, token=token)


@app.post("/auth/login", response_model=AuthResponse)
def login(body: LoginRequest):
    conn = get_conn()
    try:
        user = get_user(conn, body.user_id)
        if not user or not verify_password(body.password, user["password_hash"], user["password_salt"]):
            raise HTTPException(401, "invalid user_id or password")
        token = create_session(conn, user["user_id"])
    finally:
        conn.close()
    return AuthResponse(user_id=user["user_id"], n_interactions=user["n_interactions"], token=token)


@app.get("/auth/session/{token}", response_model=AuthResponse)
def session_lookup(token: str):
    """Resolves a 'remember me' token (stashed in the frontend's URL query
    string, which survives a hard browser reload unlike st.session_state)
    back to a user -- lets a reloaded page restore login without a
    password."""
    conn = get_conn()
    try:
        user = get_session_user(conn, token)
    finally:
        conn.close()
    if not user:
        raise HTTPException(404, "session not found")
    return AuthResponse(user_id=user["user_id"], n_interactions=user["n_interactions"], token=token)


@app.delete("/auth/session/{token}")
def session_logout(token: str):
    conn = get_conn()
    try:
        delete_session(conn, token)
    finally:
        conn.close()
    return {"status": "ok"}


# ---------------------------------------------------------------- behavior events + cart
@app.post("/events/{user_id}/{item_id}")
def record_event(user_id: str, item_id: str, event_type: str = "view"):
    """Logs a view/like/cart-add as a real interaction row -- the very next
    /recommend call for this user re-reads history fresh from Postgres, so
    this is what makes recommendations change from live behavior without
    retraining anything (see api/db.py log_event's docstring)."""
    if event_type not in ("view", "like", "cart"):
        raise HTTPException(400, "event_type must be one of: view, like, cart")
    conn = get_conn()
    try:
        log_event(conn, user_id, item_id, event_type)
    finally:
        conn.close()
    return {"status": "ok"}


@app.post("/cart/{user_id}/{item_id}")
def cart_add(user_id: str, item_id: str):
    conn = get_conn()
    try:
        add_to_cart(conn, user_id, item_id)
    finally:
        conn.close()
    return {"status": "ok"}


@app.delete("/cart/{user_id}/{item_id}")
def cart_remove(user_id: str, item_id: str):
    conn = get_conn()
    try:
        remove_from_cart(conn, user_id, item_id)
    finally:
        conn.close()
    return {"status": "ok"}


@app.get("/cart/{user_id}", response_model=CartResponse)
def cart_view(user_id: str):
    conn = get_conn()
    try:
        items = get_cart(conn, user_id)
    finally:
        conn.close()
    return CartResponse(items=[Item(**it) for it in items])


@app.post("/checkout/{user_id}", response_model=CartResponse)
def cart_checkout(user_id: str):
    """'Buy Now': no real payment (this is a demo), but logs every cart item
    as a purchase -- the strongest positive signal -- and empties the cart."""
    conn = get_conn()
    try:
        items = checkout(conn, user_id)
    finally:
        conn.close()
    return CartResponse(items=[Item(**it) for it in items])


# ---------------------------------------------------------------- wishlist
@app.post("/wishlist/{user_id}/{item_id}")
def wishlist_add(user_id: str, item_id: str):
    conn = get_conn()
    try:
        add_to_wishlist(conn, user_id, item_id)
    finally:
        conn.close()
    return {"status": "ok"}


@app.delete("/wishlist/{user_id}/{item_id}")
def wishlist_remove(user_id: str, item_id: str):
    conn = get_conn()
    try:
        remove_from_wishlist(conn, user_id, item_id)
    finally:
        conn.close()
    return {"status": "ok"}


@app.get("/wishlist/{user_id}", response_model=CartResponse)
def wishlist_view(user_id: str):
    conn = get_conn()
    try:
        items = get_wishlist(conn, user_id)
    finally:
        conn.close()
    return CartResponse(items=[Item(**it) for it in items])


# ---------------------------------------------------------------- order history
@app.get("/orders/{user_id}", response_model=OrdersResponse)
def orders_view(user_id: str):
    conn = get_conn()
    try:
        rows = get_orders(conn, user_id)
    finally:
        conn.close()
    for r in rows:
        r["purchased_at"] = str(r["purchased_at"])
    return OrdersResponse(items=[OrderItem(**r) for r in rows])
