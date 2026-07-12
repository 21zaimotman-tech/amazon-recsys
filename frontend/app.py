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
import random
from urllib.parse import quote
import requests
import streamlit as st
from streamlit_searchbox import st_searchbox

API = os.getenv("API_URL", "http://api:8000")
st.set_page_config(page_title="ElectroPicks", page_icon="⚡", layout="wide")

# Client mode by default: model names, latency pills, and the diversity
# slider are engineering telemetry, not shopping UI. Append ?debug=1 to the
# URL to get them back (e.g. for the defense / demo walkthrough).
DEBUG = str(st.query_params.get("debug", "")).lower() in ("1", "true", "yes")

# Browser auto-translate (and some extensions, e.g. Grammarly) rewrite text
# nodes in place (<font> wrappers), which crashes Streamlit's React tree
# with "NotFoundError: removeChild" error boxes on rerun. Mark the whole
# document notranslate so translators leave the DOM alone. The component
# iframe is same-origin, so its script can reach the parent document.
import streamlit.components.v1 as _components
_components.html(
    """<script>
    const d = window.parent.document;
    d.documentElement.setAttribute('translate', 'no');
    d.body.classList.add('notranslate');
    if (!d.querySelector('meta[name="google"][content="notranslate"]')) {
        const m = d.createElement('meta');
        m.name = 'google'; m.content = 'notranslate';
        d.head.appendChild(m);
    }
    </script>""",
    height=0,
)

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

/* ---- brand ---- */
.ep-wordmark {
    font-size: 1.6rem; font-weight: 800; color: #111827; letter-spacing: -0.02em;
}
.ep-wordmark span {
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.ep-tagline { font-size: 0.72rem; color: #6B7280; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; margin-top: -0.2rem; }

/* ---- utility bar (value props, real-shop dressing) ---- */
.ep-topbar {
    background: linear-gradient(90deg, #1E3A8A, #4F46E5 45%, #7C3AED);
    color: #E0E7FF; font-size: 0.74rem; font-weight: 600; text-align: center;
    padding: 0.4rem 0.8rem; border-radius: 10px; margin-bottom: 0.9rem;
    letter-spacing: 0.02em;
}

/* ---- hero (logged-out home) ---- */
.ep-hero {
    background: linear-gradient(120deg, #EEF2FF, #F5F3FF 55%, #ECFEFF);
    border: 1px solid #E0E7FF; border-radius: 16px;
    padding: 1.4rem 1.6rem; margin: 0.4rem 0 1.1rem 0;
}
.ep-hero-title { font-size: 1.6rem; font-weight: 800; color: #111827; letter-spacing: -0.02em; }
.ep-hero-sub { color: #4B5563; font-size: 0.95rem; margin-top: 0.3rem; }

/* ---- section eyebrow + underline ---- */
.ep-eyebrow {
    font-size: 0.68rem; font-weight: 800; color: #7C3AED; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 0.1rem;
}
.ep-section-title { position: relative; padding-bottom: 0.4rem; }
.ep-section-title::after {
    content: ""; position: absolute; left: 0; bottom: 0; width: 44px; height: 3px;
    border-radius: 99px; background: linear-gradient(90deg, #2563EB, #7C3AED);
}

/* ---- star ratings (partial fill via overlay width) ---- */
.ep-stars { position: relative; display: inline-block; font-size: 0.78rem; line-height: 1; }
.ep-stars-bg { color: #E5E7EB; letter-spacing: 1px; }
.ep-stars-fg { color: #F59E0B; letter-spacing: 1px; position: absolute; left: 0; top: 0;
    overflow: hidden; white-space: nowrap; }
.ep-stars-num { font-size: 0.7rem; font-weight: 700; color: #B45309; margin-left: 4px; }

/* ---- image badge + hover zoom ---- */
.ep-media-wrap { position: relative; overflow: hidden; border-radius: 8px; }
.ep-flag {
    position: absolute; top: 8px; left: 8px; z-index: 2;
    background: linear-gradient(90deg, #F59E0B, #F97316); color: #FFFFFF;
    font-size: 0.58rem; font-weight: 800; padding: 2px 8px; border-radius: 999px;
    letter-spacing: 0.06em;
}
div[data-testid="stColumn"]:hover .ep-card-img { transform: scale(1.05); }

/* ---- footer ---- */
.ep-footer { border-top: 1px solid #E5E7EB; margin-top: 2.2rem; padding-top: 1.1rem; }
.ep-footer-props {
    display: flex; flex-wrap: wrap; gap: 1.4rem; justify-content: center;
    font-size: 0.8rem; font-weight: 600; color: #374151;
}
.ep-footer-note { text-align: center; color: #9CA3AF; font-size: 0.72rem; margin-top: 0.7rem; }

/* ---- category pills + primary buttons on-brand ---- */
div[data-testid="stPills"] button {
    border-radius: 999px !important; font-size: 0.78rem !important; font-weight: 600 !important;
}
div[data-testid="stPills"] button:hover { border-color: #7C3AED !important; color: #7C3AED !important; }
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #2563EB, #7C3AED) !important; border: none !important;
}

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
    transition: transform 220ms ease;
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
.ep-card-reason {
    font-size: 0.66rem; color: #2563EB; font-weight: 600; margin-top: 0.15rem;
    height: 0.95rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.ep-card-row { display: flex; align-items: center; justify-content: space-between; margin-top: 0.25rem; }
.ep-card-price { font-size: 0.86rem; font-weight: 700; color: #111827; }
.ep-price-na { color: #9CA3AF; font-weight: 500; }
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

/* ---- search suggestion chips (as-you-type, above the full result grid) ---- */
.ep-suggest-label { font-size: 0.72rem; font-weight: 700; color: #9CA3AF; margin: 0.4rem 0 0.3rem 0; }
div[data-testid="stElementContainer"][class*="st-key-suggest-"] div[data-testid="stButton"] button {
    border-radius: 999px !important; background: #F3F4F6 !important; border: 1px solid #E5E7EB !important;
    color: #374151 !important; font-size: 0.76rem !important; font-weight: 600 !important;
    padding: 0.3rem 0.7rem !important; white-space: nowrap !important; overflow: hidden !important;
    text-overflow: ellipsis !important;
}
div[data-testid="stElementContainer"][class*="st-key-suggest-"] div[data-testid="stButton"] button:hover {
    border-color: #2563EB !important; color: #2563EB !important; background: #EFF6FF !important;
}

/* ---- header popover triggers (user chip / Log in) ---- */
/* A long username must truncate with an ellipsis, not wrap the header
   button onto three broken lines. */
div[data-testid="stPopover"] button p {
    white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important;
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
if "session_token" not in st.session_state:
    st.session_state.session_token = None       # "remember me" token, also stashed in the URL (?t=)
                                                 # so a hard reload (new Streamlit session, wipes
                                                 # session_state) can restore login -- see _restore_session
if "viewed_items" not in st.session_state:
    st.session_state.viewed_items = set()       # de-dupes "view" event logging per browser session
if "page" not in st.session_state:
    st.session_state.page = "home"              # "home" | "wishlist" | "cart" | "orders"
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None      # {"type": "wishlist"|"cart"|"buy", "item_id": ...}
                                                 # set when a logged-out user clicks an action button --
                                                 # replayed automatically right after they log in.
if "search_query" not in st.session_state:
    st.session_state.search_query = ""          # committed full-catalog search (see the header's
                                                 # st_searchbox -> "See all results" suggestion)
if "_last_search_selection" not in st.session_state:
    st.session_state._last_search_selection = None
for _k, _v in (("n_pop", 20), ("n_rec", 20), ("n_search", 24), ("n_cat", 25)):
    if _k not in st.session_state:
        st.session_state[_k] = _v       # per-surface grid sizes, grown by "Load more"
if "_last_committed_query" not in st.session_state:
    st.session_state._last_committed_query = ""
if "_last_category" not in st.session_state:
    st.session_state._last_category = None
if "feed_seed" not in st.session_state:
    # One seed per browser session: the API's temperature sampling uses it,
    # so the feed is stable while you browse (no reshuffle on every click)
    # but fresh on every reload/new visit (new Streamlit session, new seed)
    # -- and on every logo click (see the Home handler).
    st.session_state.feed_seed = random.randint(0, 2**31 - 1)


def _goto(page):
    """Explicit navigation (header icons / Home) exits search mode too --
    otherwise a committed search keeps rendering over the Wishlist/Cart/
    Orders page the user just asked for (search wins in the routing)."""
    st.session_state.page = page
    st.session_state.selected_item = None
    st.session_state.search_query = ""
    if "item" in st.query_params:
        del st.query_params["item"]
    st.rerun()


def _select_item(item_id):
    """Also mirrors the selection into the URL (?item=...) so a product page
    is shareable/bookmarkable -- the same query-params trick the login token
    uses to survive a hard reload."""
    st.session_state.selected_item = item_id
    st.query_params["item"] = item_id
    st.rerun()


def _restore_session():
    """A hard browser reload (F5) opens a brand-new Streamlit session, which
    wipes st.session_state -- so user_id would reset to None even though
    the person never logged out. The URL's query string DOES survive a
    reload, so login/signup stash a 'remember me' token there (?t=...);
    this looks it up against the backend and silently restores the login on
    the next run. No-op once already logged in, and fails quietly (stays
    logged out) if the token is stale/invalid instead of showing an error
    banner -- an expired/missing session is an expected, not exceptional,
    outcome here."""
    if st.session_state.user_id:
        return
    token = st.query_params.get("t")
    if not token:
        return
    try:
        r = requests.get(f"{API}/auth/session/{token}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            st.session_state.user_id = data["user_id"]
            st.session_state.session_token = token
        else:
            del st.query_params["t"]
    except Exception:
        pass


_restore_session()

# Shared/bookmarked product URL (?item=...): open that product's detail
# panel on load. Close/nav handlers delete the param before rerunning, so
# this only fires on a genuine external open, not after a Close.
_shared_item = st.query_params.get("item")
if _shared_item and st.session_state.selected_item is None:
    st.session_state.selected_item = _shared_item


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


def put(path):
    try:
        r = requests.put(f"{API}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error on {path}: {e}")
        return None


def _categories():
    """Top catalog categories (browse strip + onboarding picker), fetched
    once per browser session."""
    if "categories_cache" not in st.session_state:
        data = get("/categories?n=10")
        st.session_state.categories_cache = [c["category"] for c in (data or {}).get("categories", [])]
    return st.session_state.categories_cache


def _share_url(item_id):
    """Absolute shareable link to one product (?item=... opens its detail
    panel directly). Host/scheme come from the request headers so the link
    is right both on localhost and behind the production proxy."""
    try:
        host = st.context.headers.get("Host", "localhost:8501")
        proto = st.context.headers.get("X-Forwarded-Proto", "http")
        return f"{proto}://{host}/?item={item_id}"
    except Exception:
        return f"/?item={item_id}"



def load_more_button(count_key, step, cap, btn_key, n_returned):
    """Centered 'Load more' under a grid. Hidden once the surface is
    exhausted (returned fewer than requested) or at the cap."""
    if n_returned < st.session_state[count_key] or st.session_state[count_key] >= cap:
        return
    _l, _mid, _r = st.columns([2, 1.2, 2])
    with _mid:
        if st.button("Load more ↓", key=btn_key, use_container_width=True):
            st.session_state[count_key] = min(cap, st.session_state[count_key] + step)
            st.rerun()


def render_filter_controls(keyp):
    """Shared 'Filter & sort' expander (search results + category browse).
    Returns a query-string suffix for /search or /category."""
    with st.expander("Filter & sort"):
        f1, f2, f3, f4, f5 = st.columns([1.4, 1, 1, 1, 1.2])
        sort_label = f1.selectbox("Sort by", ["Relevance", "Price: low to high",
                                              "Price: high to low", "Rating"], key=f"{keyp}_sort")
        min_r_label = f2.selectbox("Min rating", ["Any", "4.5+", "4.0+", "3.5+"], key=f"{keyp}_minr")
        pmin = f3.number_input("Min $", min_value=0.0, value=0.0, step=5.0, key=f"{keyp}_pmin")
        pmax = f4.number_input("Max $ (0 = any)", min_value=0.0, value=0.0, step=5.0, key=f"{keyp}_pmax")
        fbrand = f5.text_input("Brand", key=f"{keyp}_brand")
    sort = {"Relevance": "relevance", "Price: low to high": "price_asc",
            "Price: high to low": "price_desc", "Rating": "rating"}[sort_label]
    suffix = f"&sort={sort}"
    if min_r_label != "Any":
        suffix += f"&min_rating={min_r_label.rstrip('+')}"
    if pmin > 0:
        suffix += f"&price_min={pmin}"
    if pmax > 0:
        suffix += f"&price_max={pmax}"
    if fbrand.strip():
        suffix += f"&brand={quote(fbrand.strip())}"
    return suffix


def _latency_ms_text(latency_ms: dict) -> str:
    total = (latency_ms or {}).get("total")
    return f"{total:.0f}ms" if isinstance(total, (int, float)) else "—"


def _search_suggest(searchterm):
    """Callback for the header's live search box (streamlit-searchbox).
    Unlike a plain st.text_input -- which, as of Streamlit 1.36+, only
    commits its value and reruns the script on Enter/blur, not on every
    keystroke ("Press Enter to apply") -- this component calls back into
    Python on a debounced timer as the user types, with no Enter needed, so
    suggestions genuinely appear automatically.

    Suggestions are short query completions from /suggest ("head" ->
    "headphones", "headset"), NOT full product titles -- a type-ahead
    dropdown is a search suggester, not a result list. Picking one (or the
    trailing "See all results" entry) runs the catalog-grid search."""
    term = (searchterm or "").strip()
    if len(term) < 2:
        return []
    try:
        r = requests.get(f"{API}/suggest", params={"q": term, "n": 8}, timeout=5)
        r.raise_for_status()
        words = r.json()["suggestions"]
    except Exception:
        return []
    options = [(w, ("query", w)) for w in words if w != term]
    options.append((f'\U0001F50E See all results for "{term}"', ("query", term)))
    return options


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


_ACTION_LABEL = {
    "wishlist": "save this to your wishlist",
    "cart": "add this to your cart",
    "buy": "buy this item",
}


def _require_login(action_type, item_id):
    """Wishlist/Add to cart/Buy stay visible even logged out -- clicking one
    without an account queues the action and reruns straight into the login
    prompt (see the banner right after the header) instead of just hiding
    the buttons behind a caption. render_auth_popover() replays the queued
    action the moment login/signup succeeds."""
    st.session_state.pending_action = {"type": action_type, "item_id": item_id}
    st.rerun()


def _run_pending_action():
    pa = st.session_state.pending_action
    if not pa:
        return
    item_id = pa["item_id"]
    if pa["type"] == "wishlist":
        wishlist_add(item_id)
        st.toast("Saved to wishlist — your recommendations will reflect this now.", icon="⭐")
    elif pa["type"] == "cart":
        cart_add(item_id)
        st.toast("Added to cart.", icon="🛒")
    elif pa["type"] == "buy":
        # Single-item purchase (POST /buy) -- routing this through the cart
        # + full checkout would silently buy whatever else was already there.
        data, _, _ = post(f"/buy/{st.session_state.user_id}/{item_id}")
        if data:
            st.toast("Order placed — thanks!", icon="✅")
            st.session_state.page = "orders"
    st.session_state.pending_action = None


def _stars_html(rating):
    """Real 5-star row with partial fill (gold overlay clipped to rating%),
    instead of a '⭐ 4.5' text blob -- reads like an actual shop."""
    if not rating:
        return ""
    pct = max(0.0, min(100.0, rating / 5 * 100))
    return (f'<span class="ep-stars"><span class="ep-stars-bg">★★★★★</span>'
            f'<span class="ep-stars-fg" style="width:{pct:.0f}%">★★★★★</span></span>'
            f'<span class="ep-stars-num">{rating:.1f}</span>')


def _item_media_html(it, css_class="ep-card-placeholder", size_style=""):
    if it.get("image_url"):
        return f'<img class="{css_class}" style="{size_style}" src="{it["image_url"]}" />'
    bg, _ = _category_tint(it.get("category") or "")
    icon = _category_icon(it.get("category"), it.get("title") or "")
    return f'<div class="{css_class}" style="{size_style}background:{bg};">{icon}</div>'


def _complete_login(data):
    """Shared by login and signup: sets the session, stashes the 'remember
    me' token in the URL so a hard reload survives (see _restore_session),
    then replays whatever action (wishlist/cart/buy) triggered the login
    prompt in the first place. A brand-new account (no interactions yet)
    gets the category onboarding picker on the home page."""
    st.session_state.user_id = data["user_id"]
    st.session_state.session_token = data.get("token")
    if st.session_state.session_token:
        st.query_params["t"] = st.session_state.session_token
    st.session_state.viewed_items = set()
    st.session_state.page = "home"
    st.session_state.needs_onboarding = (data.get("n_interactions", 0) == 0)
    _run_pending_action()
    st.rerun()


def render_auth_popover(key_suffix=""):
    """key_suffix keeps widget keys unique across call sites -- Streamlit
    renders st.popover content on every rerun regardless of whether the
    popover is open (only visibility is toggled client-side), and now this
    is called both from the header popover AND the inline login banner
    (see _require_login) when both happen to be present on the same run, so
    two calls with identical keys would collide (duplicate st.form key)."""
    pa = st.session_state.pending_action
    if pa:
        st.info(f"Log in or sign up to {_ACTION_LABEL.get(pa['type'], 'continue')}.")
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])
    with tab_login:
        with st.form(f"login-form{key_suffix}", border=False):
            u = st.text_input("Username", key=f"login-user{key_suffix}")
            p = st.text_input("Password", type="password", key=f"login-pass{key_suffix}")
            if st.form_submit_button("Log in", use_container_width=True):
                data, status, detail = post("/auth/login", {"user_id": u, "password": p},
                                            quiet_errors=(401, 429))
                if data:
                    _complete_login(data)
                elif status == 401:
                    st.error("Wrong username or password.")
                elif status == 429:
                    st.error("Too many failed attempts — try again in a few minutes.")
    with tab_signup:
        with st.form(f"signup-form{key_suffix}", border=False):
            u = st.text_input("Choose a username", key=f"signup-user{key_suffix}")
            p = st.text_input("Choose a password", type="password", key=f"signup-pass{key_suffix}")
            if st.form_submit_button("Create account", use_container_width=True):
                data, status, detail = post("/auth/register", {"user_id": u, "password": p},
                                            quiet_errors=(409, 400))
                if data:
                    _complete_login(data)
                elif status == 409:
                    st.error(f"'{u}' is already taken.")
                elif status == 400:
                    st.error("Username and password are both required.")


def render_item_row(it, key_prefix, actions, extra_caption=None):
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
        # \$ -- two bare $ in one markdown string trigger KaTeX math mode
        # (price + line total turned the whole caption italic)
        price = f"\\${it['price']:.2f}" if it.get("price") else ""
        rating = f"⭐ {it['avg_rating']:.1f}" if it.get("avg_rating") else ""
        st.markdown(f"**{title}**")
        st.caption(" · ".join(filter(None, [meta, price, rating, extra_caption])))
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
            _select_item(item_id)

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
        qty = it.get("qty") or 1

        def _view(item_id=item_id):
            _select_item(item_id)

        def _dec(item_id=item_id, qty=qty):
            put(f"/cart/{st.session_state.user_id}/{item_id}?qty={qty - 1}")  # 0 removes the line
            st.rerun()

        def _inc(item_id=item_id, qty=qty):
            put(f"/cart/{st.session_state.user_id}/{item_id}?qty={qty + 1}")
            st.rerun()

        def _remove(item_id=item_id):
            cart_remove(item_id)
            st.rerun()

        line_total = (it.get("price") or 0) * qty
        extra = f"qty {qty}" + (f" · line total \\${line_total:.2f}" if line_total else "")
        render_item_row(it, "cartrow", [("−", _dec), ("＋", _inc), ("View", _view), ("Remove", _remove)],
                        extra_caption=extra)
        st.divider()
    total = sum((it.get("price") or 0) * (it.get("qty") or 1) for it in items)
    if total:
        st.markdown(f"**Total: ${total:.2f}**", unsafe_allow_html=True)
    if st.button("✅ Buy now", key="buy-now-page", use_container_width=True, type="primary"):
        bought = checkout()
        if bought:
            n = sum((it.get("qty") or 1) for it in bought["items"])
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
        qty = it.get("qty") or 1
        cols = st.columns([1, 3.2, 1.4])
        with cols[0]:
            st.markdown(_item_media_html(it, css_class="ep-row-thumb"), unsafe_allow_html=True)
        with cols[1]:
            title = (it.get("title") or it.get("item_id") or "")[:80]
            price = f"\\${it['price']:.2f}" if it.get("price") else ""
            st.markdown(f"**{title}**")
            st.caption(" · ".join(filter(None, [price, f"qty {qty}" if qty > 1 else ""])))
        with cols[2]:
            purchased_at = (it.get("purchased_at") or "")[:10]
            st.caption(f"Purchased {purchased_at}" if purchased_at else "")
        st.divider()
    total = sum((it.get("price") or 0) * (it.get("qty") or 1) for it in items)
    if total:
        st.markdown(f"**Total spent: ${total:.2f}**")


# ---------------------------------------------------------------- header
st.markdown('<div class="ep-topbar">🚚 Free shipping over $50 &nbsp;·&nbsp; ↩️ 30-day returns '
            '&nbsp;·&nbsp; 🔒 Secure checkout &nbsp;·&nbsp; ⚡ Picks tuned to you in real time</div>',
            unsafe_allow_html=True)
head_l, head_c, head_r = st.columns([2, 3, 3.2])
with head_l:
    st.markdown('<div class="ep-wordmark">Electro<span>Picks</span></div>'
                '<div class="ep-tagline">Tech picks, tuned to you</div>', unsafe_allow_html=True)
    # Invisible full-column button, same overlay technique as product cards
    # (see the stColumn/stButton CSS above) -- clicking the logo area resets
    # to the home page: clears the open detail panel and the committed
    # search (the search box's own leftover text is harmless -- only
    # search_query controls whether the results section renders).
    if st.button("Home", key="cardview-home-nav"):
        # Clicking the store logo means "take me home, fresh": new feed
        # seed -> the API resamples, so new products appear (same behavior
        # as a hard reload). Other nav (wishlist/cart) keeps the seed, so
        # the feed doesn't reshuffle under the user mid-browse.
        st.session_state.feed_seed = random.randint(0, 2**31 - 1)
        st.session_state.search_query = ""
        _goto("home")
with head_c:
    _selection = st_searchbox(
        _search_suggest,
        key="search_box",
        placeholder="Search the whole catalog…",
        clear_on_submit=True,
        label="Search",
    )
    # st_searchbox keeps returning the same selected value on every rerun
    # until the user picks something new -- without this guard, acting on
    # it here would re-trigger _select_item()/rerun() forever.
    if _selection and _selection != st.session_state._last_search_selection:
        st.session_state._last_search_selection = _selection
        _kind, _value = _selection
        if _kind == "item":
            _select_item(_value)
        else:
            st.session_state.search_query = _value
            st.rerun()
search_query = st.session_state.search_query
with head_r:
    if st.session_state.user_id:
        wish_col, cart_col, orders_col, user_col = st.columns([1, 1, 1, 1.9])
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
                    if st.session_state.session_token:
                        delete(f"/auth/session/{st.session_state.session_token}")
                    if "t" in st.query_params:
                        del st.query_params["t"]
                    st.session_state.user_id = None
                    st.session_state.session_token = None
                    st.session_state.selected_item = None
                    st.session_state.page = "home"
                    st.rerun()
    else:
        with st.popover("Log in / Sign up", use_container_width=True):
            render_auth_popover(key_suffix="-header")

st.divider()

# Wishlist/Add to cart/Buy stay clickable for logged-out visitors (see
# _require_login) -- this banner is where that login actually happens,
# right at the top of the page regardless of whether the click came from a
# grid card or the detail panel, and login/signup here replays the queued
# action automatically (render_auth_popover -> _run_pending_action).
if st.session_state.pending_action and not st.session_state.user_id:
    with st.container(border=True):
        render_auth_popover(key_suffix="-banner")

with st.sidebar:
    st.markdown('<div class="ep-wordmark" style="font-size:1.25rem;">Electro<span>Picks</span></div>'
                '<div class="ep-tagline">Tech picks, tuned to you</div>', unsafe_allow_html=True)
    if DEBUG:
        st.caption("A recsys demo: different pages, different models — on purpose.")
        temp = st.slider("Diversity (temperature)", 0.0, 3.0, 1.0, 0.1)
        st.caption("Results resample as you adjust this — the API applies temperature sampling server-side.")
    else:
        st.caption("Electronics picks, personalized to you.")
        temp = 1.0

    # Client-facing variety control (MMR re-ranking) -- visible for everyone,
    # applied to both the trending and the personalized feed. The UI uses a
    # friendly 1-3 scale; MMR's lambda is mathematically a 0-1 mixing weight
    # (lam*relevance - (1-lam)*similarity), so we map: 1 -> lam=1.0 (off),
    # 2 -> 0.75, 3 -> 0.5 (max variety).
    st.markdown("**Feed variety**")
    variety = st.slider("Variety", 1.0, 3.0, 1.0, 0.25, label_visibility="collapsed",
                        help="Maximal Marginal Relevance re-ranking: each pick trades relevance "
                             "against similarity to items already picked. 1 = pure relevance, "
                             "3 = maximum variety.")
    mmr_lam = round(1.0 - (variety - 1.0) * 0.25, 3)
    st.caption("Drag right for more varied picks — fewer near-identical products.")

    if DEBUG and st.button("📊 Analytics", use_container_width=True):
        st.session_state.page = "analytics"
        st.session_state.selected_item = None
        st.rerun()
    if st.session_state.user_id:
        st.success(f"Browsing as `{st.session_state.user_id}`")
        if DEBUG:
            st.caption("Clicking, liking, and adding items to your cart all feed back into your "
                      "recommendations on the next page load — no retraining, the model just reads "
                      "your latest activity fresh every time.")
        else:
            st.caption("What you view, save, and buy tunes your picks instantly.")
        st.divider()
        st.markdown("**Your stuff**")
        _orders_n = len((get(f"/orders/{st.session_state.user_id}") or {}).get("items", []))
        if st.button(f"♡ Wishlist ({n_wish})", use_container_width=True, key="side-wishlist"):
            _goto("wishlist")
        if st.button(f"🛒 Cart ({n_cart})", use_container_width=True, key="side-cart"):
            _goto("cart")
        if st.button(f"📦 Orders ({_orders_n})", use_container_width=True, key="side-orders"):
            _goto("orders")
    else:
        st.info("Log in (or sign up) to get picks that adapt to what you view, save, and buy.")
    st.divider()
    st.markdown("**Browse**")
    for _c in _categories()[:6]:
        if st.button(_c, use_container_width=True, key=f"side-cat-{_c}"):
            # inject the pick into the browse strip's widget state (set
            # BEFORE the widget instantiates this run, which is allowed),
            # then route home where the category grid renders
            _pill_key = "category_pills" if hasattr(st, "pills") else "category_select"
            st.session_state[_pill_key] = _c
            st.session_state.page = "home"
            st.session_state.search_query = ""
            st.session_state.selected_item = None
            if "item" in st.query_params:
                del st.query_params["item"]
            st.rerun()
    st.divider()
    st.caption("Recommendations adapt in real time to what you view, save, and buy.")

# ---------------------------------------------------------------- card renderers
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
                # A missing price renders as a muted em-dash, not an empty
                # gap -- blank space where every other card shows a price
                # reads as broken to a shopper.
                price_html = (f'<span class="ep-card-price">${it["price"]:.2f}</span>' if it.get("price")
                              else '<span class="ep-card-price ep-price-na">—</span>')
                rating_html = _stars_html(it.get("avg_rating"))
                if it.get("image_url"):
                    media = f'<img class="ep-card-img" src="{it["image_url"]}" />'
                else:
                    bg, _ = _category_tint(it.get("category") or "")
                    icon = _category_icon(it.get("category"), title)
                    media = f'<div class="ep-card-placeholder" style="background:{bg};">{icon}</div>'
                # honest merchandising: flag only what the data supports
                flag = '<span class="ep-flag">TOP RATED</span>' if (it.get("avg_rating") or 0) >= 4.7 else ""
                media = f'<div class="ep-media-wrap">{flag}{media}</div>'
                brand_html = f'<div class="ep-card-brand">{brand}</div>' if brand else ""
                badge_html = (f'<span class="ep-card-badge" style="background:{tint_bg};color:{tint_fg};">'
                              f'{model_label}</span>' if DEBUG else "")
                reason_html = (f'<div class="ep-card-reason">{it["reason"][:60]}</div>'
                               if it.get("reason") else "")
                card_html = (
                    f'<div class="ep-card">{media}{brand_html}<div class="ep-card-title">{title}</div>'
                    f'<div class="ep-card-meta">{meta}</div>'
                    f'{reason_html}'
                    f'<div class="ep-card-row">{price_html}'
                    f'<span class="ep-card-rating">{rating_html}</span></div>'
                    f'{badge_html}'
                    f'<div class="ep-card-hint">View details →</div></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("View", key=f"cardview-{key_prefix}-{it['item_id']}"):
                    _select_item(it["item_id"])
                # Visible even logged out -- clicking it without an account
                # queues the add and routes into the login banner instead of
                # just being hidden (see _require_login).
                if st.button("🛒 Add", key=f"cardcart-{key_prefix}-{it['item_id']}", help="Add to cart"):
                    if st.session_state.user_id:
                        cart_add(it["item_id"])
                        st.toast("Added to cart.", icon="🛒")
                        st.rerun()
                    else:
                        _require_login("cart", it["item_id"])


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


def render_analytics_page():
    """Debug-mode dashboard over the live interaction log."""
    st.markdown('<div class="ep-section-title">Analytics</div>', unsafe_allow_html=True)
    data = get("/admin/stats")
    if not data:
        return
    import pandas as pd
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Users", data["users"])
    m2.metric("Active (24h)", data["active_24h"])
    m3.metric("Orders", data["orders"])
    m4.metric("Revenue", f"${data['revenue']:.2f}")
    if data.get("events_by_type"):
        st.markdown("**Live events by type**")
        st.bar_chart(pd.Series(data["events_by_type"]))
    col_a, col_b = st.columns(2)
    with col_a:
        if data.get("top_viewed"):
            st.markdown("**Most viewed products**")
            st.dataframe(pd.DataFrame(data["top_viewed"]), use_container_width=True, hide_index=True)
    with col_b:
        if data.get("top_categories"):
            st.markdown("**Most engaged categories**")
            st.dataframe(pd.DataFrame(data["top_categories"]), use_container_width=True, hide_index=True)


def model_caption(model_label, latency_ms):
    """Which model served this section -- ALWAYS shown (the course brief
    requires displaying which model produced each recommendation). The
    latency pill is engineering telemetry and stays debug-only.

    Single-line HTML on purpose: with the latency span empty, a multi-line
    block leaves a whitespace-only line, CommonMark ends the HTML block
    there, and the orphaned </div> renders as text while the unclosed <div>
    swallows sibling nodes -- which then crashes React reconciliation with
    removeChild errors on the cards below."""
    latency_html = (f'<span class="ep-latency-pill">{_latency_ms_text(latency_ms)}</span>'
                    if DEBUG else "")
    st.markdown(
        f'<div class="ep-model-caption"><span class="ep-model-pill">model: {model_label}</span>'
        f'{latency_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- detail panel
def render_detail_panel():
    item_id = st.session_state.selected_item
    if not item_id:
        return
    it = st.session_state.item_cache.get(item_id)
    if not it or not it.get("title"):
        # Opened from a shared URL (?item=...) with nothing browsed yet --
        # fetch the metadata directly; quietly drop bad/stale links.
        try:
            r = requests.get(f"{API}/item/{item_id}", timeout=10)
        except Exception:
            r = None
        if r is not None and r.status_code == 200:
            it = r.json()
            st.session_state.item_cache[item_id] = it
        else:
            st.session_state.selected_item = None
            if "item" in st.query_params:
                del st.query_params["item"]
            st.warning("That product link doesn't exist (anymore).")
            return
    log_view(item_id)   # counts as behavioral signal -- see log_view's docstring
    title = it.get("title") or item_id
    stars = _stars_html(it.get("avg_rating"))
    meta = " · ".join(filter(None, [it.get("category"), it.get("brand")]))
    price = f"${it['price']:.2f}" if it.get("price") else ""

    if it.get("image_url"):
        media = f'<img class="ep-detail-img" src="{it["image_url"]}" />'
    else:
        bg, _ = _category_tint(it.get("category") or "")
        icon = _category_icon(it.get("category"), title)
        media = f'<div class="ep-detail-placeholder" style="background:{bg};">{icon}</div>'
    meta_html = f'<div class="ep-detail-meta">{meta}</div>' if meta else ""
    price_html = (f'<div class="ep-detail-price">{price}</div>' if price
                  else '<div class="ep-detail-meta">Price unavailable</div>')

    # Everything visual in one HTML block, on a single line: Streamlit renders
    # each separate st.markdown()/st.button() call as its own top-level element
    # (an opening <div> in one call and content in the next never actually
    # nest), and a multi-line block risks a blank line wherever an optional
    # field like meta_html/price_html is empty -- CommonMark ends an HTML
    # block at the first blank line, so anything after would render as a
    # literal (indented-code-block) text dump instead of HTML.
    stars_html = f'<div style="margin-top:0.35rem;">{stars}</div>' if stars else ""
    detail_html = (
        f'<div class="ep-detail"><div class="ep-detail-grid">{media}'
        f'<div><div class="ep-detail-title">{title}</div>{meta_html}{stars_html}{price_html}</div>'
        f'</div></div>'
    )
    st.markdown(detail_html, unsafe_allow_html=True)

    # Wishlist/Add to cart/Buy now stay visible even logged out -- clicking
    # one without an account queues it and routes into the login banner
    # right under the header instead of hiding the buttons behind a caption
    # (see _require_login / _run_pending_action).
    close_col, wish_col, cart_col, buy_col, share_col = st.columns([1, 1.1, 1.3, 1.2, 1])
    with close_col:
        if st.button("✕ Close", key="close-detail"):
            st.session_state.selected_item = None
            if "item" in st.query_params:      # before rerun, or the shared-URL
                del st.query_params["item"]    # restore would re-open it
            st.rerun()
    with share_col:
        with st.popover("🔗 Share", use_container_width=True):
            st.caption("Anyone with this link lands straight on this product:")
            st.code(_share_url(item_id), language=None)
    # header nav (wishlist/cart counts) renders earlier in the script than
    # these handlers on THIS pass, so it'd still show pre-click counts
    # until the st.rerun() below re-executes top to bottom.
    with wish_col:
        if st.button("☆ Wishlist", key=f"wish-{item_id}"):
            if st.session_state.user_id:
                wishlist_add(item_id)
                st.toast("Saved to wishlist — your recommendations will reflect this now.", icon="⭐")
                st.rerun()
            else:
                _require_login("wishlist", item_id)
    with cart_col:
        if st.button("🛒 Add to cart", key=f"cart-{item_id}"):
            if st.session_state.user_id:
                cart_add(item_id)
                st.toast("Added to cart.", icon="🛒")
                st.rerun()
            else:
                _require_login("cart", item_id)
    with buy_col:
        if st.button("⚡ Buy now", key=f"buy-{item_id}"):
            if st.session_state.user_id:
                # Buys THIS item only (POST /buy). Never goes through
                # /checkout, which purchases and clears the entire cart.
                data, _, _ = post(f"/buy/{st.session_state.user_id}/{item_id}")
                if data:
                    st.toast("Order placed — thanks!", icon="✅")
                    _goto("orders")
            else:
                _require_login("buy", item_id)

    st.markdown("**Similar items**")
    sim = get(f"/similar/{item_id}?n=10")
    if sim:
        render_hscroll(sim["items"], key_prefix="sim")


# ---------------------------------------------------------------- main content
if st.session_state.selected_item:
    render_detail_panel()

# Category browse strip -- home page only, and only when a search isn't
# already narrowing things. Picking a pill swaps the feed below for that
# category's top-rated grid; picking it again toggles back to the feed.
_active_category = None
if not search_query.strip() and st.session_state.page == "home":
    _cats = _categories()
    if _cats:
        if hasattr(st, "pills"):
            _active_category = st.pills("Browse categories", _cats, key="category_pills",
                                        label_visibility="collapsed")
        else:                              # older Streamlit: selectbox fallback
            _sel = st.selectbox("Browse categories", ["All categories"] + _cats, key="category_select")
            _active_category = None if _sel == "All categories" else _sel

# Search always wins when non-empty, even on a Wishlist/Cart/Orders page --
# then the logged-in-only management pages (page state persists until Home
# or a nav button is clicked), then category browse, then the home feed.
if search_query.strip():
    # Real catalog search via GET /search, not a filter of whatever ~20
    # items happened to already be on the page -- a term that exists in the
    # catalog but not in today's recommendations should still find something.
    st.markdown(f'<div class="ep-section-title">Search results for &quot;{search_query}&quot;</div>',
               unsafe_allow_html=True)
    if st.session_state._last_committed_query != search_query:
        st.session_state._last_committed_query = search_query
        st.session_state.n_search = 24          # new query -> reset grid size
    _filters = render_filter_controls("s")
    data = get(f"/search?q={quote(search_query.strip())}&n={st.session_state.n_search}{_filters}")
    if data:
        items = data["items"]
        if not items:
            st.caption("No items match your search and filters.")
        else:
            cache_items(items)
            suggestions = items[:6]
            if len(suggestions) > 1:
                st.markdown('<div class="ep-suggest-label">TOP MATCHES</div>', unsafe_allow_html=True)
                sugg_cols = st.columns(len(suggestions))
                for col, it in zip(sugg_cols, suggestions):
                    with col:
                        label = (it.get("title") or it.get("brand") or it.get("item_id"))[:26]
                        if st.button(label, key=f"suggest-{it['item_id']}", use_container_width=True,
                                    help=it.get("title")):
                            _select_item(it["item_id"])
            render_grid(items, "Search", key_prefix="search")
            load_more_button("n_search", 24, 96, "more-search", len(items))
elif st.session_state.page == "wishlist" and st.session_state.user_id:
    render_wishlist_page()
elif st.session_state.page == "cart" and st.session_state.user_id:
    render_cart_page()
elif st.session_state.page == "orders" and st.session_state.user_id:
    render_orders_page()
elif st.session_state.page == "analytics" and DEBUG:
    render_analytics_page()
elif _active_category:
    st.markdown(f'<div class="ep-section-title">{_active_category}</div>', unsafe_allow_html=True)
    if st.session_state._last_category != _active_category:
        st.session_state._last_category = _active_category
        st.session_state.n_cat = 25             # new category -> reset grid size
    _filters = render_filter_controls("c")
    data = get(f"/category/{quote(_active_category)}?n={st.session_state.n_cat}{_filters}")
    if data:
        model_caption(data["model_label"], data["latency_ms"])
        render_grid(data["items"], data["model_label"], key_prefix="cat")
        load_more_button("n_cat", 25, 100, "more-cat", len(data["items"]))
elif not st.session_state.user_id:
    st.markdown('<div class="ep-hero"><div class="ep-hero-title">Find your next favorite gadget ⚡</div>'
                '<div class="ep-hero-sub">Trending picks refresh on every visit — log in and they '
                'start learning what <em>you</em> love.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="ep-eyebrow">What everyone\'s buying</div>'
                '<div class="ep-section-title">Trending electronics</div>', unsafe_allow_html=True)
    # Keyed by feed seed: a logo click / reload changes the seed, and the new
    # key REMOUNTS this whole subtree instead of patching nodes in place --
    # browser extensions that mutate the DOM (Grammarly-style scanners)
    # otherwise make React's reconciliation crash with removeChild errors.
    with st.container(key=f"popfeed-{st.session_state.feed_seed}-{st.session_state.n_pop}"):
        data = get(f"/popular?n={st.session_state.n_pop}&seed={st.session_state.feed_seed}"
                   f"&mmr_lambda={mmr_lam}")
        if data:
            model_caption(data["model_label"], data["latency_ms"])
            render_grid(data["items"], data["model_label"], key_prefix="pop")
            load_more_button("n_pop", 20, 60, "more-pop", len(data["items"]))
else:
    # Cold-start onboarding: a brand-new account has nothing for the model
    # to read, so the first feed would be generic popularity. Two "like"
    # seeds per picked category personalize it immediately.
    if st.session_state.get("needs_onboarding"):
        with st.container(border=True):
            st.markdown("**Welcome! What are you into?**")
            st.caption("Pick a few categories and your feed personalizes right away — "
                       "or skip and it'll learn from what you click.")
            _chosen = st.multiselect("Categories", _categories(), key="onboard_cats",
                                     label_visibility="collapsed", placeholder="Choose categories…")
            ob1, ob2, _sp = st.columns([1.2, 1, 3.5])
            if ob1.button("Personalize my feed", type="primary", disabled=not _chosen):
                for _cat in _chosen[:5]:
                    _top = get(f"/category/{quote(_cat)}?n=2")
                    for _it in (_top or {}).get("items", []):
                        requests.post(f"{API}/events/{st.session_state.user_id}/{_it['item_id']}"
                                      f"?event_type=like", timeout=10)
                st.session_state.needs_onboarding = False
                st.toast("Got it — your feed is now personalized.", icon="✨")
                st.rerun()
            if ob2.button("Skip for now"):
                st.session_state.needs_onboarding = False
                st.rerun()

    # Recently viewed comes FIRST -- picking up where you left off is the
    # most common intent on a return visit.
    recent = get(f"/recent/{st.session_state.user_id}?n=10")
    if recent and recent["items"]:
        st.markdown('<div style="font-weight:700;font-size:1.05rem;">Keep browsing</div>',
                    unsafe_allow_html=True)
        render_hscroll(recent["items"], key_prefix="recent")

    st.markdown('<div class="ep-eyebrow" style="margin-top:0.8rem;">Picked for you</div>'
                '<div class="ep-section-title">For you</div>',
                unsafe_allow_html=True)
    # Seed-keyed remount, same reasoning as the trending feed above.
    with st.container(key=f"recfeed-{st.session_state.feed_seed}-{st.session_state.n_rec}"):
        data = get(f"/recommend/{st.session_state.user_id}?n={st.session_state.n_rec}"
                   f"&temperature={temp}"
                   f"&seed={st.session_state.feed_seed}&mmr_lambda={mmr_lam}")
        if data:
            model_caption(data["model_label"], data["latency_ms"])
            render_grid(data["items"], data["model_label"], key_prefix="rec")
            load_more_button("n_rec", 20, 50, "more-rec", len(data["items"]))

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

# ---------------------------------------------------------------- footer
st.markdown(
    '<div class="ep-footer">'
    '<div class="ep-footer-props"><span>🚚 Free shipping over $50</span>'
    '<span>↩️ 30-day returns</span><span>🔒 Secure checkout</span>'
    '<span>💬 24/7 support</span></div>'
    '<div class="ep-footer-note">© 2026 ElectroPicks · Recommendations adapt in real time '
    'to what you view, save, and buy.</div></div>',
    unsafe_allow_html=True,
)
