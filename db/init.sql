-- Postgres schema: items metadata, user interaction history (test-period),
-- and optional user profiles. Loaded once by scripts/populate_db.py.

-- trigram matching: ILIKE '%term%' full-scans without it -- fine at ~10K
-- items, sluggish at the full ~160K catalog. GIN+pg_trgm keeps search and
-- type-ahead suggestions fast at any catalog size.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS items (
    item_id    TEXT PRIMARY KEY,
    title      TEXT,
    image_url  TEXT,
    category   TEXT,
    brand      TEXT,
    price      REAL,
    avg_rating REAL
);
CREATE INDEX IF NOT EXISTS idx_items_title_trgm    ON items USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_items_brand_trgm    ON items USING gin (brand gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_items_category_trgm ON items USING gin (category gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_items_category      ON items (category);

-- interactions used as USER HISTORY at serving time = test-period activity
-- (the model serves on this but was trained only on the train period).
CREATE TABLE IF NOT EXISTS interactions (
    user_id    TEXT,
    item_id    TEXT REFERENCES items(item_id),
    rating     REAL,
    ts         BIGINT,       -- ms since epoch
    event_type TEXT          -- view/like/cart/wishlist/purchase; NULL on
                             -- rows imported from the offline dataset
);
CREATE INDEX IF NOT EXISTS idx_inter_user ON interactions(user_id, ts DESC);

CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT PRIMARY KEY,
    password_hash  TEXT NOT NULL,
    password_salt  TEXT NOT NULL,
    n_interactions INT NOT NULL DEFAULT 0,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- opaque "remember me" tokens so login survives a hard browser reload (a
-- full page reload starts a brand-new Streamlit session, which wipes
-- st.session_state -- the frontend stashes this token in the URL's query
-- string, which DOES survive a reload, and re-resolves it back to a
-- user_id here on the next run). Not a full auth session system (no
-- expiry) -- fine for a demo, but a real product would want one.
CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    TEXT REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- persistent cart, separate from the interaction-log signal below (a user's
-- cart is what they intend to buy right now; interactions are the full
-- engagement history the model reads at serving time)
CREATE TABLE IF NOT EXISTS cart (
    user_id  TEXT REFERENCES users(user_id) ON DELETE CASCADE,
    item_id  TEXT REFERENCES items(item_id),
    qty      INTEGER NOT NULL DEFAULT 1,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, item_id)
);

-- "save for later" / liked items, distinct from cart (intent to buy now).
-- Liking an item both logs a "like" interaction event AND adds it here --
-- one user action, one place to browse it back.
CREATE TABLE IF NOT EXISTS wishlist (
    user_id  TEXT REFERENCES users(user_id) ON DELETE CASCADE,
    item_id  TEXT REFERENCES items(item_id),
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, item_id)
);

-- permanent purchase record, written at checkout ("Buy now") and never
-- deleted -- unlike cart (current intent, cleared on checkout) this is the
-- user's order history. No primary key on (user_id,item_id): the same item
-- can be bought more than once, each a separate row/order.
CREATE TABLE IF NOT EXISTS orders (
    order_id      SERIAL PRIMARY KEY,
    user_id       TEXT REFERENCES users(user_id) ON DELETE CASCADE,
    item_id       TEXT REFERENCES items(item_id),
    qty           INTEGER NOT NULL DEFAULT 1,
    price         REAL,             -- unit price at purchase time, not looked up later
    purchased_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id, purchased_at DESC);
