"""Streamlit frontend. Calls the FastAPI service and renders item cards with
images. Each section uses a DIFFERENT model, just like a real product:

  Homepage (logged out)  -> /popular            (Popularity)
  Homepage (logged in)   -> /recommend          (Two-tower -> LightGBM)
  "Because you liked X"  -> /because-you-liked   (embedding cosine)
  Item page similar      -> /similar             (embedding cosine)

Visual/UX layer only — every API call and the get() error-handling pattern
below is unchanged from the original. No new backend endpoints.
"""
import os
from urllib.parse import quote
import requests
import streamlit as st

API = os.getenv("API_URL", "http://api:8000")
st.set_page_config(page_title="ElectroPicks", page_icon="⚡", layout="wide")

# ---------------------------------------------------------------- category styling
# A small fixed palette so the grid reads as organized (same category -> same
# tint every time) rather than one random color per product.
_PALETTE = [
    ("#EFF6FF", "#2563EB"),  # blue
    ("#F0FDF4", "#16A34A"),  # green
    ("#FDF4FF", "#A21CAF"),  # purple
    ("#FFF7ED", "#C2410C"),  # amber
    ("#FFF1F2", "#BE123C"),  # rose
    ("#F0FDFA", "#0D9488"),  # teal
    ("#FEFCE8", "#A16207"),  # yellow
    ("#F5F3FF", "#6D28D9"),  # violet
]

_ICON_RULES = [
    (("headphone", "earbud", "earphone"), "🎧"),
    (("camera", "webcam"), "📷"),
    (("laptop", "notebook", "macbook", "chromebook"), "💻"),
    (("watch",), "⌚"),
    (("bulb", "light"), "💡"),
    (("keyboard",), "⌨️"),
    (("speaker", "soundbar", "audio"), "🔊"),
    (("phone", "mobile", "iphone"), "📱"),
    (("cable", "charger", "adapter", "power"), "🔌"),
    (("tv", "monitor", "display", "screen"), "📺"),
    (("mouse",), "🖱️"),
    (("tablet", "ipad"), "📱"),
    (("drive", "storage", "ssd", "usb"), "💾"),
    (("router", "network", "wifi"), "📡"),
    (("battery",), "🔋"),
    (("case", "cover", "bag"), "🎒"),
    (("cable", "cord", "wire"), "🔌"),
]


def _category_tint(category: str) -> tuple[str, str]:
    key = category or "General"
    idx = sum(ord(c) for c in key) % len(_PALETTE)
    return _PALETTE[idx]


def _category_icon(category: str, title: str) -> str:
    haystack = f"{category or ''} {title or ''}".lower()
    for keywords, icon in _ICON_RULES:
        if any(k in haystack for k in keywords):
            return icon
    return "📦"


# ---------------------------------------------------------------- CSS
st.markdown(
    """
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1200px;}
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
}
body { background: #FAFAFA; }

/* ---- header ---- */
.ep-wordmark {
    font-size: 1.5rem; font-weight: 800; color: #111827; letter-spacing: -0.02em;
}
.ep-wordmark span { color: #2563EB; }

/* ---- section heading ---- */
.ep-section-title { font-size: 1.75rem; font-weight: 800; color: #111827; margin-bottom: 0.15rem; }
.ep-model-caption {
    display: inline-flex; align-items: center; gap: 0.4rem;
    color: #6B7280; font-size: 0.85rem; margin-bottom: 1.1rem;
}
.ep-model-pill {
    background: #EFF6FF; color: #2563EB; font-weight: 600; font-size: 0.72rem;
    padding: 0.15rem 0.55rem; border-radius: 999px; letter-spacing: 0.01em;
}
.ep-latency-pill {
    background: #F3F4F6; color: #374151; font-weight: 600; font-size: 0.72rem;
    padding: 0.15rem 0.55rem; border-radius: 999px;
}

/* ---- product card ---- */
.ep-card {
    background: #FFFFFF; border: 1px solid #EEF0F3; border-radius: 12px;
    padding: 0.7rem 0.7rem 0.85rem 0.7rem; margin-bottom: 0.6rem;
    transition: transform 150ms ease, box-shadow 150ms ease;
    height: 100%;
}
/* the click target is an invisible button layered on top of (a sibling of,
   not a descendant of) .ep-card, so :hover on the card itself never fires --
   the shared column ancestor is what actually receives the pointer. */
div[data-testid="stColumn"]:hover .ep-card {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(17, 24, 39, 0.08);
    border-color: #E5E7EB;
}
.ep-card-img {
    width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 8px;
    background: #F9FAFB; display: block;
}
.ep-card-placeholder {
    width: 100%; aspect-ratio: 1 / 1; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 2.6rem;
}
.ep-card-brand {
    font-size: 0.66rem; font-weight: 700; color: #9CA3AF; margin-top: 0.55rem;
    text-transform: uppercase; letter-spacing: 0.04em;
}
.ep-card-title {
    font-size: 0.86rem; font-weight: 600; color: #111827; margin-top: 0.2rem;
    line-height: 1.25rem; height: 2.5rem; overflow: hidden;
}
.ep-card-meta { font-size: 0.75rem; color: #6B7280; margin-top: 0.15rem; }
.ep-card-row { display: flex; align-items: center; justify-content: space-between; margin-top: 0.25rem; }
.ep-card-price { font-size: 0.86rem; font-weight: 700; color: #111827; }
.ep-card-rating { font-size: 0.72rem; font-weight: 700; color: #B45309; }
.ep-card-badge {
    display: inline-block; font-size: 0.65rem; font-weight: 700; margin-top: 0.4rem;
    padding: 0.1rem 0.45rem; border-radius: 999px;
}

/* The whole card is clickable: an invisible real Streamlit button is
   absolutely positioned over the entire column, on top of the pure-HTML
   card underneath it, so any click on the card (not just a small "View"
   button) opens the detail panel while the click target stays a real
   Streamlit widget (raw HTML can't trigger a Python callback).

   IMPORTANT: this must be scoped to buttons keyed "cardview-*" specifically
   (via the [class*="st-key-cardview-"] qualifier below), not every button
   inside every stColumn -- an earlier, broader version of this rule made
   the detail panel's Close/Like/Wishlist/Add-to-cart buttons invisible too,
   since those also happen to live inside st.columns(). They existed and
   were clickable (Playwright found them fine) but opacity:0 hid them from
   view entirely -- a real bug, caught by screenshot, not by any exception. */
div[data-testid="stColumn"] { position: relative !important; }
/* Streamlit gives a button's own stElementContainer position:relative by
   default, making IT (not stColumn) the containing block for our absolutely
   positioned button -- and since that container collapses to height:0 (its
   only child is now absolutely positioned, contributing no intrinsic
   height back to it), inset:0 resolves against a zero-height box. Force it
   back to static so stColumn is the containing block instead -- but again,
   only for cardview-keyed buttons, not globally. */
div[data-testid="stColumn"] div[data-testid="stElementContainer"][class*="st-key-cardview-"] {
    position: static !important;
}
div[data-testid="stColumn"] div[data-testid="stElementContainer"][class*="st-key-cardview-"]
    div[data-testid="stButton"] {
    position: absolute !important;
    top: 0 !important; left: 0 !important; right: 0 !important; bottom: 0 !important;
    width: 100% !important; height: 100% !important;
    z-index: 5 !important; margin: 0 !important;
}
div[data-testid="stColumn"] div[data-testid="stElementContainer"][class*="st-key-cardview-"]
    div[data-testid="stButton"] button {
    width: 100% !important; height: 100% !important;
    opacity: 0 !important; cursor: pointer; border: none !important; background: transparent !important;
}
.ep-card-hint {
    font-size: 0.68rem; color: #2563EB; font-weight: 600; margin-top: 0.35rem; opacity: 0;
    transition: opacity 150ms ease;
}
div[data-testid="stColumn"]:hover .ep-card-hint { opacity: 1; }

/* Quick "add to cart" button on each card: a real, VISIBLE button in a
   small bottom-right corner zone with a higher z-index (6) than the
   full-card "view details" overlay (5) above, so a click there is
   captured by this button instead of falling through to the card-wide
   one. Selectors here stack one extra attribute qualifier on top of the
   general column/button rules above specifically so they win on
   specificity regardless of source order (both use !important). */
div[data-testid="stColumn"] div[data-testid="stElementContainer"][class*="st-key-cardcart-"] {
    position: absolute !important; z-index: 6 !important;
    top: auto !important; left: auto !important; right: 8px !important; bottom: 8px !important;
    width: 78px !important; height: 30px !important; margin: 0 !important;
}
div[data-testid="stColumn"] div[data-testid="stElementContainer"][class*="st-key-cardcart-"]
    div[data-testid="stButton"] {
    position: static !important; width: 78px !important; height: 30px !important;
}
div[data-testid="stColumn"] div[data-testid="stElementContainer"][class*="st-key-cardcart-"]
    div[data-testid="stButton"] button {
    opacity: 1 !important; width: 78px !important; height: 30px !important;
    border-radius: 999px !important; background: #2563EB !important; color: #FFFFFF !important;
    border: none !important; font-size: 0.72rem !important; font-weight: 700 !important; padding: 0 !important;
    box-shadow: 0 2px 6px rgba(37,99,235,0.45);
}
div[data-testid="stColumn"] div[data-testid="stElementContainer"][class*="st-key-cardcart-"]
    div[data-testid="stButton"] button:hover { background: #1D4ED8 !important; }

/* ---- horizontal scroll rows (Because you liked / Similar items) ---- */
div[class*="st-key-hscroll-"] div[data-testid="stHorizontalBlock"] {
    overflow-x: auto; flex-wrap: nowrap; padding-bottom: 0.6rem; gap: 0.75rem;
}
div[class*="st-key-hscroll-"] div[data-testid="stHorizontalBlock"]::-webkit-scrollbar { height: 6px; }
div[class*="st-key-hscroll-"] div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb {
    background: #E5E7EB; border-radius: 999px;
}
div[class*="st-key-hscroll-"] div[data-testid="stColumn"] { min-width: 130px; flex: 0 0 130px; }
.ep-hcard {
    background: #FFFFFF; border: 1px solid #EEF0F3; border-radius: 12px;
    padding: 0.5rem; transition: transform 150ms ease, box-shadow 150ms ease;
}
div[class*="st-key-hscroll-"] div[data-testid="stColumn"]:hover .ep-hcard {
    transform: translateY(-3px); box-shadow: 0 6px 14px rgba(17,24,39,0.07);
}
.ep-hcard-img { width: 100%; aspect-ratio: 1/1; object-fit: cover; border-radius: 8px; }
.ep-hcard-placeholder {
    width: 100%; aspect-ratio: 1/1; border-radius: 8px; font-size: 1.8rem;
    display: flex; align-items: center; justify-content: center;
}
.ep-hcard-title { font-size: 0.72rem; font-weight: 600; color: #111827; margin-top: 0.4rem;
    line-height: 1.05rem; height: 2.1rem; overflow: hidden; }

/* ---- detail panel ---- */
.ep-detail {
    background: #FFFFFF; border: 1px solid #EEF0F3; border-radius: 14px;
    padding: 1.1rem 1.3rem; margin: 0.6rem 0 0.6rem 0;
}
.ep-detail-grid { display: flex; gap: 1.3rem; align-items: flex-start; }
.ep-detail-img { width: 220px; height: 220px; object-fit: cover; border-radius: 10px; flex-shrink: 0; }
.ep-detail-placeholder {
    width: 220px; height: 220px; border-radius: 10px; flex-shrink: 0; font-size: 4rem;
    display: flex; align-items: center; justify-content: center;
}
.ep-detail-title { font-size: 1.15rem; font-weight: 800; color: #111827; }
.ep-detail-meta { color: #6B7280; font-size: 0.85rem; margin-top: 0.2rem; }
.ep-detail-price { font-size: 1.2rem; font-weight: 800; color: #2563EB; margin-top: 0.5rem; }

/* ---- wishlist / cart / orders row lists (management pages, not browsing --
   plain st.columns() per row instead of the card-grid overlay pattern, so
   no extra z-index layering is needed here) ---- */
.ep-row-thumb {
    width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 1.7rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- state
if "item_cache" not in st.session_state:
    st.session_state.item_cache = {}
if "selected_item" not in st.session_state:
    st.session_state.selected_item = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None            # set only after a real /auth/login or /auth/register
if "viewed_items" not in st.session_state:
    st.session_state.viewed_items = set()       # de-dupes "view" event logging per browser session
if "page" not in st.session_state:
    st.session_state.page = "home"              # "home" | "wishlist" | "cart" | "orders"


def _goto(page):
    st.session_state.page = page
    st.session_state.selected_item = None
    st.rerun()


def cache_items(items):
    for it in items:
        st.session_state.item_cache[it["item_id"]] = it


def get(path):
    try:
        r = requests.get(f"{API}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error on {path}: {e}")
        return None


def post(path, json_body=None, quiet_errors=()):
    """POST/DELETE helper mirroring get()'s error-handling pattern.
    quiet_errors: HTTP status codes to surface as a return value instead of
    an st.error banner (e.g. 401 on login -- that's an expected outcome of
    a bad password, not an API failure worth alarming the whole page over)."""
    try:
        r = requests.post(f"{API}{path}", json=json_body, timeout=10)
        if r.status_code in quiet_errors:
            return None, r.status_code, r.json().get("detail", "")
        r.raise_for_status()
        return r.json(), r.status_code, None
    except requests.HTTPError as e:
        st.error(f"API error on {path}: {e}")
        return None, e.response.status_code if e.response is not None else None, None
    except Exception as e:
        st.error(f"API error on {path}: {e}")
        return None, None, None


def delete(path):
    try:
        r = requests.delete(f"{API}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error on {path}: {e}")
        return None


def _latency_ms_text(latency_ms: dict) -> str:
    total = (latency_ms or {}).get("total")
    return f"{total:.0f}ms" if isinstance(total, (int, float)) else "—"


# ---------------------------------------------------------------- accounts + events
def log_view(item_id):
    """Once per item per browser session -- avoid re-logging a "view" on
    every rerun the detail panel stays open for (e.g. clicking Wishlist)."""
    if st.session_state.user_id and item_id not in st.session_state.viewed_items:
        requests.post(f"{API}/events/{st.session_state.user_id}/{item_id}?event_type=view", timeout=10)
        st.session_state.viewed_items.add(item_id)


def cart_add(item_id):
    requests.post(f"{API}/cart/{st.session_state.user_id}/{item_id}", timeout=10)


def cart_remove(item_id):
    delete(f"/cart/{st.session_state.user_id}/{item_id}")


def wishlist_add(item_id):
    """Liking an item IS saving it to the wishlist -- one user action, one
    place to find it again, instead of a separate "Like" that went nowhere
    browsable. POST /wishlist already logs a "wishlist" interaction event
    server-side (api/db.py add_to_wishlist), so this alone is both signals."""
    requests.post(f"{API}/wishlist/{st.session_state.user_id}/{item_id}", timeout=10)


def wishlist_remove(item_id):
    delete(f"/wishlist/{st.session_state.user_id}/{item_id}")


def checkout():
    data, status, detail = post(f"/checkout/{st.session_state.user_id}")
    return data


def _item_media_html(it, css_class="ep-card-placeholder", size_style=""):
    if it.get("image_url"):
        return f'<img class="{css_class}" style="{size_style}" src="{it["image_url"]}" />'
    bg, _ = _category_tint(it.get("category") or "")
    icon = _category_icon(it.get("category"), it.get("title") or "")
    return f'<div class="{css_class}" style="{size_style}background:{bg};">{icon}</div>'


def render_auth_popover():
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])
    with tab_login:
        with st.form("login-form", border=False):
            u = st.text_input("Username", key="login-user")
            p = st.text_input("Password", type="password", key="login-pass")
            if st.form_submit_button("Log in", use_container_width=True):
                data, status, detail = post("/auth/login", {"user_id": u, "password": p},
                                            quiet_errors=(401,))
                if data:
                    st.session_state.user_id = data["user_id"]
                    st.session_state.viewed_items = set()
                    st.session_state.page = "home"
                    st.rerun()
                elif status == 401:
                    st.error("Wrong username or password.")
    with tab_signup:
        with st.form("signup-form", border=False):
            u = st.text_input("Choose a username", key="signup-user")
            p = st.text_input("Choose a password", type="password", key="signup-pass")
            if st.form_submit_button("Create account", use_container_width=True):
                data, status, detail = post("/auth/register", {"user_id": u, "password": p},
                                            quiet_errors=(409, 400))
                if data:
                    st.session_state.user_id = data["user_id"]
                    st.session_state.viewed_items = set()
                    st.session_state.page = "home"
                    st.rerun()
                elif status == 409:
                    st.error(f"'{u}' is already taken.")
                elif status == 400:
                    st.error("Username and password are both required.")


def render_item_row(it, key_prefix, actions):
    """One row per item on a management page (Wishlist/Cart) -- real
    st.columns() side by side, unlike the browse grid's card-overlay trick,
    since these are simple list rows, not a tall grid needing a full-card
    click target. `actions` is a list of (label, callback) pairs, each
    rendered as its own button column."""
    cols = st.columns([1, 3.2] + [1.1] * len(actions))
    with cols[0]:
        st.markdown(_item_media_html(it, css_class="ep-row-thumb"), unsafe_allow_html=True)
    with cols[1]:
        title = (it.get("title") or it.get("item_id") or "")[:80]
        meta = " · ".join(filter(None, [it.get("category"), it.get("brand")]))
        price = f"${it['price']:.2f}" if it.get("price") else ""
        rating = f"⭐ {it['avg_rating']:.1f}" if it.get("avg_rating") else ""
        st.markdown(f"**{title}**")
        st.caption(" · ".join(filter(None, [meta, price, rating])))
    for (label, cb), col in zip(actions, cols[2:]):
        with col:
            if st.button(label, key=f"{key_prefix}-{it['item_id']}-{label}", use_container_width=True):
                cb()


def render_wishlist_page():
    st.markdown('<div class="ep-section-title">Your wishlist</div>', unsafe_allow_html=True)
    data = get(f"/wishlist/{st.session_state.user_id}")
    items = (data or {}).get("items", [])
    cache_items(items)
    if not items:
        st.caption("Nothing saved yet — tap ☆ Wishlist on any product to add it here.")
        return
    for it in items:
        item_id = it["item_id"]

        def _view(item_id=item_id):
            st.session_state.selected_item = item_id
            st.rerun()

        def _move(item_id=item_id):
            cart_add(item_id)
            wishlist_remove(item_id)
            st.toast("Moved to cart.", icon="🛒")
            st.rerun()

        def _remove(item_id=item_id):
            wishlist_remove(item_id)
            st.rerun()

        render_item_row(it, "wishrow", [("View", _view), ("Move to cart", _move), ("Remove", _remove)])
        st.divider()


def render_cart_page():
    st.markdown('<div class="ep-section-title">Your cart</div>', unsafe_allow_html=True)
    data = get(f"/cart/{st.session_state.user_id}")
    items = (data or {}).get("items", [])
    cache_items(items)
    if not items:
        st.caption("Your cart is empty — add items from Home or search.")
        return
    for it in items:
        item_id = it["item_id"]

        def _view(item_id=item_id):
            st.session_state.selected_item = item_id
            st.rerun()

        def _remove(item_id=item_id):
            cart_remove(item_id)
            st.rerun()

        render_item_row(it, "cartrow", [("View", _view), ("Remove", _remove)])
        st.divider()
    total = sum(it["price"] for it in items if it.get("price"))
    if total:
        st.markdown(f"**Total: ${total:.2f}**", unsafe_allow_html=True)
    if st.button("✅ Buy now", key="buy-now-page", use_container_width=True, type="primary"):
        bought = checkout()
        if bought:
            n = len(bought["items"])
            st.toast(f"Order placed — {n} item{'s' if n != 1 else ''}. Thanks!", icon="✅")
            _goto("orders")


def render_orders_page():
    st.markdown('<div class="ep-section-title">Order history</div>', unsafe_allow_html=True)
    data = get(f"/orders/{st.session_state.user_id}")
    items = (data or {}).get("items", [])
    if not items:
        st.caption("No orders yet — items you buy will show up here.")
        return
    for it in items:
        cols = st.columns([1, 3.2, 1.4])
        with cols[0]:
            st.markdown(_item_media_html(it, css_class="ep-row-thumb"), unsafe_allow_html=True)
        with cols[1]:
            title = (it.get("title") or it.get("item_id") or "")[:80]
            price = f"${it['price']:.2f}" if it.get("price") else ""
            st.markdown(f"**{title}**")
            st.caption(price)
        with cols[2]:
            purchased_at = (it.get("purchased_at") or "")[:10]
            st.caption(f"Purchased {purchased_at}" if purchased_at else "")
        st.divider()


# ---------------------------------------------------------------- header
head_l, head_c, head_r = st.columns([2, 3, 3.2])
with head_l:
    st.markdown('<div class="ep-wordmark">Electro<span>Picks</span></div>', unsafe_allow_html=True)
    # Invisible full-column button, same overlay technique as product cards
    # (see the stColumn/stButton CSS above) -- clicking the logo area resets
    # to the home page: clears the open detail panel and the search box.
    if st.button("Home", key="cardview-home-nav"):
        st.session_state["search_box"] = ""
        _goto("home")
with head_c:
    search_query = st.text_input(
        "Search", "", placeholder="Search the whole catalog…", label_visibility="collapsed",
        key="search_box",
    )
with head_r:
    if st.session_state.user_id:
        wish_col, cart_col, orders_col, user_col = st.columns(4)
        wishlist_data = get(f"/wishlist/{st.session_state.user_id}")
        n_wish = len(wishlist_data["items"]) if wishlist_data else 0
        cart_data = get(f"/cart/{st.session_state.user_id}")
        n_cart = len(cart_data["items"]) if cart_data else 0
        with wish_col:
            label = f"♡ {n_wish}" if n_wish else "♡"
            if st.button(label, key="navbtn-wishlist", use_container_width=True, help="Wishlist",
                        type="primary" if st.session_state.page == "wishlist" else "secondary"):
                _goto("wishlist")
        with cart_col:
            label = f"🛒 {n_cart}" if n_cart else "🛒"
            if st.button(label, key="navbtn-cart", use_container_width=True, help="Cart",
                        type="primary" if st.session_state.page == "cart" else "secondary"):
                _goto("cart")
        with orders_col:
            if st.button("📦", key="navbtn-orders", use_container_width=True, help="Order history",
                        type="primary" if st.session_state.page == "orders" else "secondary"):
                _goto("orders")
        with user_col:
            uid_short = st.session_state.user_id[:10] + ("…" if len(st.session_state.user_id) > 10 else "")
            with st.popover(f"👤 {uid_short}", use_container_width=True):
                st.write(f"Logged in as **{st.session_state.user_id}**")
                if st.button("Log out", use_container_width=True):
                    st.session_state.user_id = None
                    st.session_state.selected_item = None
                    st.session_state.page = "home"
                    st.rerun()
    else:
        with st.popover("Log in / Sign up", use_container_width=True):
            render_auth_popover()

st.divider()

with st.sidebar:
    st.markdown("### ElectroPicks")
    st.caption("A recsys demo: different pages, different models — on purpose.")
    temp = st.slider("Diversity (temperature)", 0.0, 3.0, 1.0, 0.1)
    st.caption("Results resample as you adjust this — the API applies temperature sampling server-side.")
    if st.session_state.user_id:
        st.success(f"Browsing as `{st.session_state.user_id}`")
        st.caption("Clicking, liking, and adding items to your cart all feed back into your "
                  "recommendations on the next page load — no retraining, the model just reads "
                  "your latest activity fresh every time.")
    else:
        st.info("Logged out — showing trending items to everyone. Log in (or sign up) to get "
               "recommendations that adapt to what you click, like, and add to cart.")

# ---------------------------------------------------------------- card renderers
def _select_item(item_id):
    st.session_state.selected_item = item_id
    st.rerun()


def render_grid(items, model_label, key_prefix, cols_n=5):
    """Interactive product grid: each card is pure HTML/CSS, with a real but
    invisible Streamlit button layered on top of the whole column (see the
    CSS above) so clicking anywhere on the card -- not just a small "View"
    label -- opens the detail panel. Raw HTML can't trigger a Python
    callback, so the click target has to stay a real Streamlit widget."""
    cache_items(items)
    shown = items
    if not shown:
        st.caption("No results to show.")
        return
    tint_bg, tint_fg = _category_tint(model_label)
    rows = [shown[i:i + cols_n] for i in range(0, len(shown), cols_n)]
    for row in rows:
        cols = st.columns(cols_n)
        for col, it in zip(cols, row):
            with col:
                title = (it.get("title") or it.get("item_id"))[:70]
                meta = it.get("category") or ""
                brand = it.get("brand") or ""
                price = f"${it['price']:.2f}" if it.get("price") else ""
                rating = f"⭐ {it['avg_rating']:.1f}" if it.get("avg_rating") else ""
                if it.get("image_url"):
                    media = f'<img class="ep-card-img" src="{it["image_url"]}" />'
                else:
                    bg, _ = _category_tint(it.get("category") or "")
                    icon = _category_icon(it.get("category"), title)
                    media = f'<div class="ep-card-placeholder" style="background:{bg};">{icon}</div>'
                brand_html = f'<div class="ep-card-brand">{brand}</div>' if brand else ""
                card_html = (
                    f'<div class="ep-card">{media}{brand_html}<div class="ep-card-title">{title}</div>'
                    f'<div class="ep-card-meta">{meta}</div>'
                    f'<div class="ep-card-row"><span class="ep-card-price">{price}</span>'
                    f'<span class="ep-card-rating">{rating}</span></div>'
                    f'<span class="ep-card-badge" style="background:{tint_bg};color:{tint_fg};">{model_label}</span>'
                    f'<div class="ep-card-hint">View details →</div></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("View", key=f"cardview-{key_prefix}-{it['item_id']}"):
                    _select_item(it["item_id"])
                if st.session_state.user_id:
                    if st.button("🛒 Add", key=f"cardcart-{key_prefix}-{it['item_id']}", help="Add to cart"):
                        cart_add(it["item_id"])
                        st.toast("Added to cart.", icon="🛒")
                        st.rerun()


def render_hscroll(items, key_prefix, empty_caption="Nothing to show yet.", cols_n=7):
    """Horizontal-scrolling row, e.g. Because you liked / Similar items.
    Same invisible-full-column-button click pattern as render_grid, just
    inside a keyed container so CSS can force real Streamlit columns to
    scroll horizontally (st.columns has no native overflow behavior)."""
    cache_items(items)
    if not items:
        st.caption(empty_caption)
        return
    with st.container(key=f"hscroll-{key_prefix}"):
        cols = st.columns(max(len(items), cols_n))
        for col, it in zip(cols, items):
            with col:
                title = (it.get("title") or it.get("item_id"))[:40]
                if it.get("image_url"):
                    media = f'<img class="ep-hcard-img" src="{it["image_url"]}" />'
                else:
                    bg, _ = _category_tint(it.get("category") or "")
                    icon = _category_icon(it.get("category"), title)
                    media = f'<div class="ep-hcard-placeholder" style="background:{bg};">{icon}</div>'
                card_html = f'<div class="ep-hcard">{media}<div class="ep-hcard-title">{title}</div></div>'
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("View", key=f"cardview-{key_prefix}-h-{it['item_id']}"):
                    _select_item(it["item_id"])


def model_caption(model_label, latency_ms):
    st.markdown(
        f"""<div class="ep-model-caption">
            <span class="ep-model-pill">model: {model_label}</span>
            <span class="ep-latency-pill">{_latency_ms_text(latency_ms)}</span>
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- detail panel
def render_detail_panel():
    item_id = st.session_state.selected_item
    if not item_id:
        return
    log_view(item_id)   # counts as behavioral signal -- see log_view's docstring
    it = st.session_state.item_cache.get(item_id, {"item_id": item_id})
    title = it.get("title") or item_id
    rating = f"⭐ {it['avg_rating']:.1f}" if it.get("avg_rating") else ""
    meta = " · ".join(filter(None, [it.get("category"), it.get("brand"), rating]))
    price = f"${it['price']:.2f}" if it.get("price") else ""

    if it.get("image_url"):
        media = f'<img class="ep-detail-img" src="{it["image_url"]}" />'
    else:
        bg, _ = _category_tint(it.get("category") or "")
        icon = _category_icon(it.get("category"), title)
        media = f'<div class="ep-detail-placeholder" style="background:{bg};">{icon}</div>'
    meta_html = f'<div class="ep-detail-meta">{meta}</div>' if meta else ""
    price_html = f'<div class="ep-detail-price">{price}</div>' if price else ""

    # Everything visual in one HTML block, on a single line: Streamlit renders
    # each separate st.markdown()/st.button() call as its own top-level element
    # (an opening <div> in one call and content in the next never actually
    # nest), and a multi-line block risks a blank line wherever an optional
    # field like meta_html/price_html is empty -- CommonMark ends an HTML
    # block at the first blank line, so anything after would render as a
    # literal (indented-code-block) text dump instead of HTML.
    detail_html = (
        f'<div class="ep-detail"><div class="ep-detail-grid">{media}'
        f'<div><div class="ep-detail-title">{title}</div>{meta_html}{price_html}</div>'
        f'</div></div>'
    )
    st.markdown(detail_html, unsafe_allow_html=True)

    close_col, wish_col, cart_col = st.columns([1, 1.2, 1.4])
    with close_col:
        if st.button("✕ Close", key="close-detail"):
            st.session_state.selected_item = None
            st.rerun()
    if st.session_state.user_id:
        # header nav (wishlist/cart counts) renders earlier in the script
        # than this handler on THIS pass, so it'd still show pre-click
        # counts until the st.rerun() below re-executes top to bottom.
        with wish_col:
            if st.button("☆ Wishlist", key=f"wish-{item_id}"):
                wishlist_add(item_id)
                st.toast("Saved to wishlist — your recommendations will reflect this now.", icon="⭐")
                st.rerun()
        with cart_col:
            if st.button("🛒 Add to cart", key=f"cart-{item_id}"):
                cart_add(item_id)
                st.toast("Added to cart.", icon="🛒")
                st.rerun()
    else:
        st.caption("Log in to save or buy items.")

    st.markdown("**Similar items**")
    sim = get(f"/similar/{item_id}?n=10")
    if sim:
        render_hscroll(sim["items"], key_prefix="sim")


# ---------------------------------------------------------------- main content
if st.session_state.selected_item:
    render_detail_panel()

# Search always wins when non-empty, even on a Wishlist/Cart/Orders page --
# then the logged-in-only management pages (page state persists until Home
# or a nav button is clicked), then the home feed.
if search_query.strip():
    # Real catalog search (all 9,487 items via GET /search), not a filter of
    # whatever ~20 items happened to already be on the page -- a term that
    # exists in the catalog but not in today's recommendations should still
    # find something.
    st.markdown(f'<div class="ep-section-title">Search results for &quot;{search_query}&quot;</div>',
               unsafe_allow_html=True)
    data = get(f"/search?q={quote(search_query.strip())}&n=24")
    if data:
        if not data["items"]:
            st.caption("No items match your search.")
        else:
            render_grid(data["items"], "Search", key_prefix="search")
elif st.session_state.page == "wishlist" and st.session_state.user_id:
    render_wishlist_page()
elif st.session_state.page == "cart" and st.session_state.user_id:
    render_cart_page()
elif st.session_state.page == "orders" and st.session_state.user_id:
    render_orders_page()
elif not st.session_state.user_id:
    st.markdown('<div class="ep-section-title">Trending electronics</div>', unsafe_allow_html=True)
    data = get("/popular?n=20")
    if data:
        model_caption(data["model_label"], data["latency_ms"])
        render_grid(data["items"], data["model_label"], key_prefix="pop")
else:
    st.markdown('<div class="ep-section-title">For you</div>', unsafe_allow_html=True)
    data = get(f"/recommend/{st.session_state.user_id}?n=20&temperature={temp}")
    if data:
        model_caption(data["model_label"], data["latency_ms"])
        render_grid(data["items"], data["model_label"], key_prefix="rec")

    byl = get(f"/because-you-liked/{st.session_state.user_id}?n=10")
    if byl:
        seed_id = (byl.get("latency_ms") or {}).get("seed_item")
        seed_item = st.session_state.item_cache.get(seed_id) if seed_id else None
        seed_title = (seed_item or {}).get("title") or seed_id
        st.markdown(
            f'<div style="font-weight:700;font-size:1.05rem;margin-top:0.8rem;">'
            f'Because you liked {("“" + seed_title[:50] + "”") if seed_title else "this"}</div>',
            unsafe_allow_html=True,
        )
        render_hscroll(byl["items"], key_prefix="byl", empty_caption="No history yet for this user.")
