-- Postgres schema: items metadata, user interaction history (test-period),
-- and optional user profiles. Loaded once by scripts/populate_db.py.

CREATE TABLE IF NOT EXISTS items (
    item_id    TEXT PRIMARY KEY,
    title      TEXT,
    image_url  TEXT,
    category   TEXT,
    brand      TEXT,
    price      REAL,
    avg_rating REAL
);

-- interactions used as USER HISTORY at serving time = test-period activity
-- (the model serves on this but was trained only on the train period).
CREATE TABLE IF NOT EXISTS interactions (
    user_id TEXT,
    item_id TEXT REFERENCES items(item_id),
    rating  REAL,
    ts      BIGINT          -- ms since epoch
);
CREATE INDEX IF NOT EXISTS idx_inter_user ON interactions(user_id, ts DESC);

CREATE TABLE IF NOT EXISTS users (
    user_id   TEXT PRIMARY KEY,
    n_interactions INT
);
