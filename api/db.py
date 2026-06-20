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
