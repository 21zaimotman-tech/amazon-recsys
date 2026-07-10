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


def search_items(conn, query, limit=20):
    """Full-catalog search by title/brand/category substring -- distinct from
    the frontend's old client-side filter, which only ever searched whatever
    ~20 items happened to already be on the page."""
    like = f"%{query}%"
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT item_id, title, image_url, category, brand, price, avg_rating FROM items "
            "WHERE title ILIKE %s OR brand ILIKE %s OR category ILIKE %s "
            "ORDER BY avg_rating DESC NULLS LAST LIMIT %s",
            (like, like, like, limit))
        return [dict(r) for r in cur.fetchall()]


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
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT u.user_id, u.n_interactions FROM sessions s "
            "JOIN users u ON u.user_id = s.user_id WHERE s.token=%s", (token,))
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
            "INSERT INTO interactions (user_id, item_id, rating, ts) VALUES (%s,%s,%s,%s)",
            (user_id, item_id, rating, ts))
        cur.execute(
            "UPDATE users SET n_interactions = n_interactions + 1 WHERE user_id=%s",
            (user_id,))
    conn.commit()


# ---------------------------------------------------------------- cart
def add_to_cart(conn, user_id, item_id):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO cart (user_id, item_id) VALUES (%s,%s) "
            "ON CONFLICT (user_id, item_id) DO NOTHING", (user_id, item_id))
    conn.commit()
    log_event(conn, user_id, item_id, "cart")


def remove_from_cart(conn, user_id, item_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM cart WHERE user_id=%s AND item_id=%s", (user_id, item_id))
    conn.commit()


def get_cart(conn, user_id):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT i.item_id, i.title, i.image_url, i.category, i.brand, i.price, i.avg_rating "
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
                "INSERT INTO orders (user_id, item_id, price) VALUES (%s,%s,%s)",
                (user_id, it["item_id"], it.get("price")))
        cur.execute("DELETE FROM cart WHERE user_id=%s", (user_id,))
    conn.commit()
    for it in items:
        log_event(conn, user_id, it["item_id"], "purchase")
    return items


def get_orders(conn, user_id):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT o.item_id, o.price, o.purchased_at, i.title, i.image_url, i.category, i.brand, i.avg_rating "
            "FROM orders o JOIN items i ON i.item_id = o.item_id "
            "WHERE o.user_id=%s ORDER BY o.purchased_at DESC", (user_id,))
        return [dict(r) for r in cur.fetchall()]


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
