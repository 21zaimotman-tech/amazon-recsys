"""Postgres access. Connection settings come from env (see .env.example)."""
import os
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
    """User's TEST-PERIOD interaction history (most recent first). This is the
    activity the model serves on but was NOT trained on."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT item_id FROM interactions WHERE user_id=%s "
            "ORDER BY ts DESC LIMIT %s", (user_id, limit))
        return [r[0] for r in cur.fetchall()]

def fetch_items(conn, item_ids):
    if not item_ids:
        return {}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT item_id, title, image_url, category, brand, price "
                    "FROM items WHERE item_id = ANY(%s)", (list(item_ids),))
        return {r["item_id"]: dict(r) for r in cur.fetchall()}


def search_items(conn, query, limit=20):
    """Full-catalog search by title/brand/category substring -- distinct from
    the frontend's old client-side filter, which only ever searched whatever
    ~20 items happened to already be on the page."""
    like = f"%{query}%"
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT item_id, title, image_url, category, brand, price FROM items "
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
_EVENT_RATING = {"view": 3.0, "like": 5.0, "cart": 5.0}


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
            "SELECT i.item_id, i.title, i.image_url, i.category, i.brand, i.price "
            "FROM cart c JOIN items i ON i.item_id = c.item_id "
            "WHERE c.user_id=%s ORDER BY c.added_at DESC", (user_id,))
        return [dict(r) for r in cur.fetchall()]
