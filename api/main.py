"""FastAPI service. Loads models + FAISS at startup, serves recommendations.
Swagger UI at /docs (required by the brief)."""
import time
from collections import defaultdict, deque

import psycopg2.errors
from fastapi import FastAPI, HTTPException
from .db import (get_conn, fetch_user_history, fetch_items, search_items, suggest_words,
                 fetch_category_top, top_categories, admin_stats, create_user, get_user,
                 log_event, add_to_cart, set_cart_qty, remove_from_cart, get_cart, checkout,
                 buy_item, add_to_wishlist, remove_from_wishlist, get_wishlist, get_orders,
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


def _brand_cap(items, n, cap=2):
    """Feed diversification: at most `cap` items per brand in the top n, so
    a feed doesn't read as three Echo Dot variants in a row. Over-capped
    items backfill the tail if the cap leaves the page short."""
    out, overflow, counts = [], [], {}
    for it in items:
        b = (it.brand or "").lower()
        if not b or counts.get(b, 0) < cap:
            counts[b] = counts.get(b, 0) + 1
            out.append(it)
        else:
            overflow.append(it)
        if len(out) >= n:
            break
    out += overflow[: n - len(out)]
    return out[:n]


def _attach_reasons(conn, items, history_ids):
    """Lightweight explainability: tag each recommended item with the most
    recent history item from the same category ("Because you viewed X").
    Heuristic attribution, not the model's actual gradient -- but honest
    enough ("same category as something you engaged with") and it reads
    like a real product."""
    if not history_ids:
        return items
    hist_meta = fetch_items(conn, history_ids)
    cat_to_title = {}
    for hid in history_ids:              # most recent first; first one per category wins
        m = hist_meta.get(hid)
        if m and m.get("category") and m["category"] not in cat_to_title:
            cat_to_title[m["category"]] = (m.get("title") or hid)
    hist_set = set(history_ids)
    for it in items:
        src = cat_to_title.get(it.category)
        if src and it.item_id not in hist_set:
            it.reason = f"Because you viewed {src[:45]}"
    return items


@app.get("/health")
def health():
    return {"status": "ok", "artifacts": {k: getattr(rec, k) is not None
            for k in ["popularity", "faiss", "tower", "ranker", "sim"]}}


@app.get("/popular", response_model=RecResponse)
def popular(n: int = 10, temperature: float = 1.0, seed: int = None):
    t0 = time.perf_counter()
    ids, label = rec.popular(n * 2, temperature, seed)   # over-fetch for the brand cap
    conn = get_conn()
    try:
        items = _brand_cap(_to_items(conn, ids), n)
    finally:
        conn.close()
    return RecResponse(model_label=label, items=items,
                       latency_ms={"total": (time.perf_counter() - t0) * 1e3})


@app.get("/recommend/{user_id}", response_model=RecResponse)
def recommend(user_id: str, n: int = 10, temperature: float = 1.0, seed: int = None):
    t0 = time.perf_counter()
    conn = get_conn()
    try:
        s = time.perf_counter()
        history = fetch_user_history(conn, user_id, limit=20)
        db_ms = (time.perf_counter() - s) * 1e3
        ids, label, timings = rec.recommend(user_id, history, n=n * 2,
                                            temperature=temperature, seed=seed)
        items = _brand_cap(_to_items(conn, ids), n)      # over-fetched above for the cap
        items = _attach_reasons(conn, items, history)
    finally:
        conn.close()
    timings["db"] = db_ms
    timings["total"] = (time.perf_counter() - t0) * 1e3
    return RecResponse(model_label=label, items=items, latency_ms=timings)


@app.get("/recent/{user_id}", response_model=RecResponse)
def recently_viewed(user_id: str, n: int = 10):
    """The user's own most recent distinct items ('Keep browsing') -- straight
    from the interaction log, no model involved."""
    t0 = time.perf_counter()
    conn = get_conn()
    try:
        ids = fetch_user_history(conn, user_id, limit=n)
        items = _to_items(conn, ids)
    finally:
        conn.close()
    return RecResponse(model_label="Recently viewed", items=items,
                       latency_ms={"total": (time.perf_counter() - t0) * 1e3})


@app.get("/item/{item_id}", response_model=Item)
def item_lookup(item_id: str):
    """Single-item metadata -- lets a shared product URL (?item=...) render
    the detail panel without the item having been browsed first."""
    conn = get_conn()
    try:
        it = fetch_items(conn, [item_id]).get(item_id)
    finally:
        conn.close()
    if it is None:
        raise HTTPException(404, f"unknown item_id '{item_id}'")
    return Item(**it)


@app.get("/categories")
def categories(n: int = 12):
    """Most-stocked categories, for the homepage browse strip and the
    cold-start onboarding picker."""
    conn = get_conn()
    try:
        cats = top_categories(conn, n=n)
    finally:
        conn.close()
    return {"categories": cats}


@app.get("/category/{name}", response_model=RecResponse)
def category_browse(name: str, n: int = 24, sort: str = "rating", min_rating: float = None,
                    price_min: float = None, price_max: float = None, brand: str = None):
    """Browse-by-category grid, with the same filter/sort options as /search."""
    t0 = time.perf_counter()
    conn = get_conn()
    try:
        ids = fetch_category_top(conn, name, limit=n, sort=sort, min_rating=min_rating,
                                 price_min=price_min, price_max=price_max, brand=brand)
        items = _to_items(conn, ids)
    finally:
        conn.close()
    return RecResponse(model_label=f"Top rated in {name}", items=items,
                       latency_ms={"total": (time.perf_counter() - t0) * 1e3})


@app.get("/similar/{item_id}", response_model=RecResponse)
def similar(item_id: str, n: int = 10):
    """Embedding neighbors, guarded by category: 'Similar items' on a
    headphone must show headphones. A long-tail item's embedding is mostly
    noise, so its raw cosine neighbors drift toward globally popular items
    (Echo Dots on a headphone page) -- keep only neighbors from the seed's
    own category, then top up with that category's best-rated items."""
    t0 = time.perf_counter()
    ids, label = rec.similar(item_id, n * 3)          # over-fetch; filtered below
    conn = get_conn()
    try:
        seed_cat = (fetch_items(conn, [item_id]).get(item_id) or {}).get("category")
        if seed_cat:
            cand = fetch_items(conn, ids)
            kept = [i for i in ids if (cand.get(i) or {}).get("category") == seed_cat]
            if len(kept) < n:
                kept += fetch_category_top(conn, seed_cat, limit=n - len(kept),
                                           exclude=[item_id, *kept])
            if kept:
                ids, label = kept, "Similar items (embedding + category)"
        ids = ids[:n]
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
def search(q: str, n: int = 20, sort: str = "relevance", min_rating: float = None,
           price_min: float = None, price_max: float = None, brand: str = None):
    """Full-catalog search (title/brand/category) with optional filters and
    sorting -- distinct from the frontend's per-page filter, which only ever
    searches whatever's already loaded."""
    t0 = time.perf_counter()
    conn = get_conn()
    try:
        rows = search_items(conn, q, limit=n, sort=sort, min_rating=min_rating,
                            price_min=price_min, price_max=price_max, brand=brand)
    finally:
        conn.close()
    return RecResponse(model_label=f'Search results for "{q}"', items=[Item(**r) for r in rows],
                       latency_ms={"total": (time.perf_counter() - t0) * 1e3})


# ---------------------------------------------------------------- accounts
@app.get("/suggest")
def suggest(q: str, n: int = 8):
    """Type-ahead query completions for the search box ("head" ->
    ["headphones", "headset", ...]). Word suggestions, not product titles --
    picking one runs a normal /search."""
    conn = get_conn()
    try:
        words = suggest_words(conn, q, n=n)
    finally:
        conn.close()
    return {"suggestions": words}


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


# Login throttle: 5 failures per account in 10 minutes -> 429. In-memory,
# so it resets on API restart and doesn't share state across replicas --
# demo-grade, but it stops naive password guessing.
_FAILED_LOGINS = defaultdict(deque)
_FAIL_WINDOW_S, _FAIL_LIMIT = 600, 5


def _throttled(user_id):
    dq = _FAILED_LOGINS[user_id]
    now = time.time()
    while dq and now - dq[0] > _FAIL_WINDOW_S:
        dq.popleft()
    return len(dq) >= _FAIL_LIMIT


@app.post("/auth/login", response_model=AuthResponse)
def login(body: LoginRequest):
    if _throttled(body.user_id):
        raise HTTPException(429, "too many failed attempts — try again in a few minutes")
    conn = get_conn()
    try:
        user = get_user(conn, body.user_id)
        if not user or not verify_password(body.password, user["password_hash"], user["password_salt"]):
            _FAILED_LOGINS[body.user_id].append(time.time())
            raise HTTPException(401, "invalid user_id or password")
        _FAILED_LOGINS.pop(body.user_id, None)
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


@app.put("/cart/{user_id}/{item_id}")
def cart_set_qty(user_id: str, item_id: str, qty: int):
    """Set a cart line's quantity explicitly (the cart page's -/+ steppers).
    qty <= 0 removes the line."""
    conn = get_conn()
    try:
        set_cart_qty(conn, user_id, item_id, qty)
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
    """Cart checkout: no real payment (this is a demo), but logs every cart
    item as a purchase -- the strongest positive signal -- and empties the
    cart."""
    conn = get_conn()
    try:
        items = checkout(conn, user_id)
    finally:
        conn.close()
    return CartResponse(items=[Item(**it) for it in items])


@app.post("/buy/{user_id}/{item_id}", response_model=CartResponse)
def buy_single(user_id: str, item_id: str):
    """Single-item 'Buy now' from a product page. Leaves the cart untouched --
    distinct from /checkout, which purchases and clears the whole cart."""
    conn = get_conn()
    try:
        it = buy_item(conn, user_id, item_id)
    finally:
        conn.close()
    if it is None:
        raise HTTPException(404, f"unknown item_id '{item_id}'")
    return CartResponse(items=[Item(**it)])


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


# ---------------------------------------------------------------- admin
@app.get("/admin/stats")
def stats():
    """Aggregates for the frontend's debug-mode analytics page."""
    conn = get_conn()
    try:
        return admin_stats(conn)
    finally:
        conn.close()
