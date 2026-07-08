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
    user_id        TEXT PRIMARY KEY,
    password_hash  TEXT NOT NULL,
    password_salt  TEXT NOT NULL,
    n_interactions INT NOT NULL DEFAULT 0,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- persistent cart, separate from the interaction-log signal below (a user's
-- cart is what they intend to buy right now; interactions are the full
-- engagement history the model reads at serving time)
CREATE TABLE IF NOT EXISTS cart (
    user_id  TEXT REFERENCES users(user_id) ON DELETE CASCADE,
    item_id  TEXT REFERENCES items(item_id),
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, item_id)
);

-- "save for later", distinct from cart (intent to buy now) and from a Like
-- (a one-off feedback signal, not a browsable list)
CREATE TABLE IF NOT EXISTS wishlist (
    user_id  TEXT REFERENCES users(user_id) ON DELETE CASCADE,
    item_id  TEXT REFERENCES items(item_id),
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, item_id)
);
