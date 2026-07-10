"""Postgres access. Connection settings come from env (see .env.example)."""
import os
import secrets
import psycopg2
import psycopg2.extras

def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "recsys"),
        user=os.getenv("POSTGRES_USER", "recsys"),
        password=os.getenv("POSTGRES_PASSWORD", "recsys"),
    )

def fetch_user_history(conn, user_id, limit=20):
    """User's TEST-PERIOD interaction history (most recent DISTINCT item
    first). This is the activity the model serves on but was NOT trained on.

    DISTINCT ON item_id (keeping each item's latest timestamp) matters: a
    raw "last N interaction ROWS" query lets one item clicked/added
    repeatedly (e.g. double-tapping Add to cart, or a view + a like on the
    same item) occupy several slots of the window, diluting the user
    embedding with duplicates of one item instead of reflecting the last N
    DIFFERENT things the user actually engaged with -- e.g. a single fresh
    wishlist add got outweighed by an older item that had 10 stray rows
    from earlier testing, even though the wishlist add was the most recent
    real signal."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT item_id FROM ("
            "  SELECT DISTINCT ON (item_id) item_id, ts FROM interactions"
            "  WHERE user_id=%s ORDER BY item_id, ts DESC"
            ") dedup ORDER BY ts DESC LIMIT %s", (user_id, limit))
        return [r[0] for r in cur.fetchall()]

def fetch_items(conn, item_ids):
    if not item_ids:
        return {}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT item_id, title, image_url, category, brand, price, avg_rating "
                    "FROM items WHERE item_id = ANY(%s)", (list(item_ids),))
        return {r["item_id"]: dict(r) for r in cur.fetchall()}


_SEARCH_SORTS = {
    "relevance":  "avg_rating DESC NULLS LAST",
    "rating":     "avg_rating DESC NULLS LAST",
    "price_asc":  "price ASC NULLS LAST",
    "price_desc": "price DESC NULLS LAST",
}


def search_items(conn, query, limit=20, sort="relevance",
                 min_rating=None, price_min=None, price_max=None, brand=None):
    """Full-catalog search by title/brand/category substring -- distinct from
    the frontend's old client-side filter, which only ever searched whatever
    ~20 items happened to already be on the page.

    Each whitespace-separated term must match somewhere (title, brand, or
    category), but terms need not be adjacent or in order -- a single ILIKE
    on the whole phrase would make "sony headphones" miss
    "Sony WH-1000XM4 ... Headphones"."""
    terms = [t for t in query.split() if t]
    if not terms:
        return []
    clauses, params = [], []
    for t in terms:
        like = f"%{t}%"
        clauses.append("(title ILIKE %s OR brand ILIKE %s OR category ILIKE %s)")
        params.extend([like, like, like])
    if min_rating is not None:
        clauses.append("avg_rating >= %s"); params.append(min_rating)
    if price_min is not None:
        clauses.append("price >= %s"); params.append(price_min)
    if price_max is not None:
        clauses.append("price <= %s"); params.append(price_max)
    if brand:
        clauses.append("brand ILIKE %s"); params.append(f"%{brand}%")
    order = _SEARCH_SORTS.get(sort, _SEARCH_SORTS["relevance"])
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT item_id, title, image_url, category, brand, price, avg_rating FROM items "
            "WHERE " + " AND ".join(clauses) +
            f" ORDER BY {order} LIMIT %s",
            (*params, limit))
        return [dict(r) for r in cur.fetchall()]


def top_categories(conn, n=12):
    """Most-stocked categories, for the homepage browse strip and the
    cold-start onboarding picker."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT category, count(*) FROM items WHERE category IS NOT NULL "
            "GROUP BY category ORDER BY count(*) DESC LIMIT %s", (n,))
        return [{"category": r[0], "n_items": r[1]} for r in cur.fetchall()]


def fetch_category_top(conn, category, limit=20, exclude=(), sort="rating",
                       min_rating=None, price_min=None, price_max=None, brand=None):
    """Items in one category, rating-ordered by default. Doubles as the
    content-based fill for 'Similar items' (defaults) and as the browse-by-
    category grid, which passes the same filter/sort options as /search."""
    clauses = ["category = %s", "NOT (item_id = ANY(%s))"]
    params = [category, list(exclude)]
    if min_rating is not None:
        clauses.append("avg_rating >= %s"); params.append(min_rating)
    if price_min is not None:
        clauses.append("price >= %s"); params.append(price_min)
    if price_max is not None:
        clauses.append("price <= %s"); params.append(price_max)
    if brand:
        clauses.append("brand ILIKE %s"); params.append(f"%{brand}%")
    order = _SEARCH_SORTS.get(sort, _SEARCH_SORTS["rating"])
    with conn.cursor() as cur:
        cur.execute(
            "SELECT item_id FROM items WHERE " + " AND ".join(clauses) +
            f" ORDER BY {order} LIMIT %s",
            (*params, limit))
        return [r[0] for r in cur.fetchall()]


def suggest_words(conn, query, n=8):
    """Type-ahead completions: complete the LAST word the user is typing into
    catalog words (from titles/brands/categories), keeping any words already
    typed before it. Returns short query strings ("sony head" -> "sony
    headphones"), not product titles -- the dropdown is a search suggester,
    not a result list.

    When earlier words exist, the completion pool is restricted to items
    matching them, so a suggested query is guaranteed to have results."""
    tokens = [t for t in query.lower().split() if t]
    if not tokens:
        return []
    prefix, head_terms = tokens[-1], tokens[:-1]
    if len(prefix) < 2:
        return []
    clauses, params = [], []
    for t in head_terms:
        like = f"%{t}%"
        clauses.append("(title ILIKE %s OR brand ILIKE %s OR category ILIKE %s)")
        params.extend([like, like, like])
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            WITH pool AS (
                SELECT title, brand, category FROM items {where} LIMIT 2000
            ),
            words AS (
                SELECT regexp_split_to_table(
                    lower(coalesce(title,'') || ' ' || coalesce(brand,'') || ' ' || coalesce(category,'')),
                    '[^a-z0-9]+') AS w
                FROM pool
            )
            SELECT w FROM words
            WHERE w LIKE %s AND length(w) >= 3 AND w !~ '^[0-9]+$'
            GROUP BY w ORDER BY count(*) DESC LIMIT %s
            """,
            (*params, prefix + "%", n))
        completions = [r[0] for r in cur.fetchall()]
    head = " ".join(head_terms)
    return [f"{head} {w}".strip() for w in completions]


# ---------------------------------------------------------------- accounts
def create_user(conn, user_id, password_hash, password_salt):
    """Raises psycopg2.errors.UniqueViolation if user_id is already taken --
    the caller (api/main.py) turns that into a 409."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (user_id, password_hash, password_salt) VALUES (%s,%s,%s)",
            (user_id, password_hash, password_salt))
    conn.commit()


def create_session(conn, user_id):
    """Opaque 'remember me' token -- see db/init.sql sessions table comment."""
    token = secrets.token_urlsafe(32)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO sessions (token, user_id) VALUES (%s,%s)", (token, user_id))
    conn.commit()
    return token


def get_session_user(conn, token):
    """Tokens older than 30 days stop resolving -- an unexpiring 'remember
    me' token in a URL is a credential that never rotates."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT u.user_id, u.n_interactions FROM sessions s "
            "JOIN users u ON u.user_id = s.user_id "
            "WHERE s.token=%s AND s.created_at > now() - interval '30 days'", (token,))
        row = cur.fetchone()
        return dict(row) if row else None


def delete_session(conn, token):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE token=%s", (token,))
    conn.commit()


def get_user(conn, user_id):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT user_id, password_hash, password_salt, n_interactions "
            "FROM users WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------- behavior events
# Ratings mirror the offline training convention (src/config.py
# POSITIVE_RATING_THRESHOLD=4.0): an explicit like/cart-add is a strong
# positive (5.0, counts as "liked"), a mere detail-page view is weaker
# signal (3.0, below the positive threshold) -- but either way it lands in
# `interactions`, which fetch_user_history reads fresh on every /recommend
# call, so it changes what the (already-trained) two-tower model serves
# without needing any retraining.
_EVENT_RATING = {"view": 3.0, "like": 5.0, "cart": 5.0, "wishlist": 4.0, "purchase": 5.0}


def log_event(conn, user_id, item_id, event_type):
    import time
    rating = _EVENT_RATING.get(event_type, 3.0)
    ts = int(time.time() * 1000)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO interactions (user_id, item_id, rating, ts, event_type) "
            "VALUES (%s,%s,%s,%s,%s)",
            (user_id, item_id, rating, ts, event_type))
        cur.execute(
            "UPDATE users SET n_interactions = n_interactions + 1 WHERE user_id=%s",
            (user_id,))
    conn.commit()


# ---------------------------------------------------------------- cart
def add_to_cart(conn, user_id, item_id):
    """Adding an item already in the cart bumps its quantity by one."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO cart (user_id, item_id) VALUES (%s,%s) "
            "ON CONFLICT (user_id, item_id) DO UPDATE SET qty = cart.qty + 1",
            (user_id, item_id))
    conn.commit()
    log_event(conn, user_id, item_id, "cart")


def set_cart_qty(conn, user_id, item_id, qty):
    """Explicit quantity from the cart page's -/+ steppers; qty <= 0 removes
    the line entirely."""
    with conn.cursor() as cur:
        if qty <= 0:
            cur.execute("DELETE FROM cart WHERE user_id=%s AND item_id=%s", (user_id, item_id))
        else:
            cur.execute("UPDATE cart SET qty=%s WHERE user_id=%s AND item_id=%s",
                        (qty, user_id, item_id))
    conn.commit()


def remove_from_cart(conn, user_id, item_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM cart WHERE user_id=%s AND item_id=%s", (user_id, item_id))
    conn.commit()


def get_cart(conn, user_id):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT i.item_id, i.title, i.image_url, i.category, i.brand, i.price, i.avg_rating, c.qty "
            "FROM cart c JOIN items i ON i.item_id = c.item_id "
            "WHERE c.user_id=%s ORDER BY c.added_at DESC", (user_id,))
        return [dict(r) for r in cur.fetchall()]


def checkout(conn, user_id):
    """'Buy Now': writes a permanent orders row per cart item (order history --
    never deleted, unlike cart), logs each as a "purchase" interaction (the
    strongest positive signal, feeding straight into the next /recommend
    call), and empties the cart. No real payment processing -- this is a
    demo -- but the behavioral feedback loop and the order history are real."""
    items = get_cart(conn, user_id)
    with conn.cursor() as cur:
        for it in items:
            cur.execute(
                "INSERT INTO orders (user_id, item_id, qty, price) VALUES (%s,%s,%s,%s)",
                (user_id, it["item_id"], it.get("qty", 1), it.get("price")))
        cur.execute("DELETE FROM cart WHERE user_id=%s", (user_id,))
    conn.commit()
    for it in items:
        log_event(conn, user_id, it["item_id"], "purchase")
    return items


def buy_item(conn, user_id, item_id):
    """Single-item 'Buy now' from a product page: one orders row + one
    "purchase" interaction for THIS item only. Unlike checkout(), never
    touches the cart -- buying one product must not silently purchase (or
    clear) whatever else the user had set aside."""
    it = fetch_items(conn, [item_id]).get(item_id)
    if it is None:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO orders (user_id, item_id, qty, price) VALUES (%s,%s,1,%s)",
            (user_id, item_id, it.get("price")))
    conn.commit()
    log_event(conn, user_id, item_id, "purchase")
    return it


def get_orders(conn, user_id):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT o.item_id, o.qty, o.price, o.purchased_at, i.title, i.image_url, i.category, i.brand, i.avg_rating "
            "FROM orders o JOIN items i ON i.item_id = o.item_id "
            "WHERE o.user_id=%s ORDER BY o.purchased_at DESC", (user_id,))
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------- admin analytics
def admin_stats(conn):
    """Aggregates for the debug-mode analytics page. event_type is NULL on
    rows imported from the offline dataset, so live-behavior breakdowns
    only count rows the API itself logged."""
    import time
    day_ago_ms = int((time.time() - 86400) * 1000)
    out = {}
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM users")
        out["users"] = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT user_id) FROM interactions WHERE ts > %s", (day_ago_ms,))
        out["active_24h"] = cur.fetchone()[0]
        cur.execute("SELECT count(*), coalesce(sum(price * qty), 0) FROM orders")
        n_orders, revenue = cur.fetchone()
        out["orders"], out["revenue"] = n_orders, float(revenue)
        cur.execute(
            "SELECT event_type, count(*) FROM interactions "
            "WHERE event_type IS NOT NULL GROUP BY event_type ORDER BY count(*) DESC")
        out["events_by_type"] = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute(
            "SELECT i.title, count(*) FROM interactions x JOIN items i ON i.item_id = x.item_id "
            "WHERE x.event_type = 'view' GROUP BY i.title ORDER BY count(*) DESC LIMIT 10")
        out["top_viewed"] = [{"title": r[0], "views": r[1]} for r in cur.fetchall()]
        cur.execute(
            "SELECT i.category, count(*) FROM interactions x JOIN items i ON i.item_id = x.item_id "
            "WHERE x.event_type IS NOT NULL AND i.category IS NOT NULL "
            "GROUP BY i.category ORDER BY count(*) DESC LIMIT 10")
        out["top_categories"] = [{"category": r[0], "events": r[1]} for r in cur.fetchall()]
    return out


# ---------------------------------------------------------------- wishlist
def add_to_wishlist(conn, user_id, item_id):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO wishlist (user_id, item_id) VALUES (%s,%s) "
            "ON CONFLICT (user_id, item_id) DO NOTHING", (user_id, item_id))
    conn.commit()
    log_event(conn, user_id, item_id, "wishlist")


def remove_from_wishlist(conn, user_id, item_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM wishlist WHERE user_id=%s AND item_id=%s", (user_id, item_id))
    conn.commit()


def get_wishlist(conn, user_id):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT i.item_id, i.title, i.image_url, i.category, i.brand, i.price, i.avg_rating "
            "FROM wishlist w JOIN items i ON i.item_id = w.item_id "
            "WHERE w.user_id=%s ORDER BY w.added_at DESC", (user_id,))
        return [dict(r) for r in cur.fetchall()]
