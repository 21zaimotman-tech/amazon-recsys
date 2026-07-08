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
.ep-card-title {
    font-size: 0.86rem; font-weight: 600; color: #111827; margin-top: 0.55rem;
    line-height: 1.25rem; height: 2.5rem; overflow: hidden;
}
.ep-card-meta { font-size: 0.75rem; color: #6B7280; margin-top: 0.15rem; }
.ep-card-price { font-size: 0.82rem; font-weight: 700; color: #111827; margin-top: 0.2rem; }
.ep-card-badge {
    display: inline-block; font-size: 0.65rem; font-weight: 700; margin-top: 0.4rem;
    padding: 0.1rem 0.45rem; border-radius: 999px;
}

/* The whole card is clickable: an invisible real Streamlit button is
   absolutely positioned over the entire column, on top of the pure-HTML
   card underneath it, so any click on the card (not just a small "View"
   button) opens the detail panel while the click target stays a real
   Streamlit widget (raw HTML can't trigger a Python callback). */
div[data-testid="stColumn"] { position: relative !important; }
/* Streamlit gives the button's own stElementContainer position:relative by
   default, making IT (not stColumn) the containing block for our absolutely
   positioned button -- and since that container collapses to height:0 (its
   only child is now absolutely positioned, contributing no intrinsic
   height back to it), inset:0 resolves against a zero-height box. Force it
   back to static so stColumn is the containing block instead. */
div[data-testid="stColumn"] div[data-testid="stElementContainer"] { position: static !important; }
div[data-testid="stColumn"] div[data-testid="stButton"] {
    position: absolute !important;
    top: 0 !important; left: 0 !important; right: 0 !important; bottom: 0 !important;
    width: 100% !important; height: 100% !important;
    z-index: 5 !important; margin: 0 !important;
}
div[data-testid="stColumn"] div[data-testid="stButton"] button {
    width: 100% !important; height: 100% !important;
    opacity: 0 !important; cursor: pointer; border: none !important; background: transparent !important;
}
.ep-card-hint {
    font-size: 0.68rem; color: #2563EB; font-weight: 600; margin-top: 0.35rem; opacity: 0;
    transition: opacity 150ms ease;
}
div[data-testid="stColumn"]:hover .ep-card-hint { opacity: 1; }

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
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- state
if "item_cache" not in st.session_state:
    st.session_state.item_cache = {}
if "selected_item" not in st.session_state:
    st.session_state.selected_item = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


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


def _latency_ms_text(latency_ms: dict) -> str:
    total = (latency_ms or {}).get("total")
    return f"{total:.0f}ms" if isinstance(total, (int, float)) else "—"


# ---------------------------------------------------------------- header
head_l, head_c, head_r = st.columns([2, 3, 2])
with head_l:
    st.markdown('<div class="ep-wordmark">Electro<span>Picks</span></div>', unsafe_allow_html=True)
with head_c:
    search_query = st.text_input(
        "Search", "", placeholder="Search the current results…", label_visibility="collapsed"
    )
with head_r:
    user_id = st.text_input(
        "Log in", "", placeholder="Log in (user_id)", label_visibility="collapsed"
    )

st.divider()

with st.sidebar:
    st.markdown("### ElectroPicks")
    st.caption("A recsys demo: different pages, different models — on purpose.")
    temp = st.slider("Diversity (temperature)", 0.0, 3.0, 1.0, 0.1)
    st.caption("Results resample as you adjust this — the API applies temperature sampling server-side.")
    if user_id:
        st.success(f"Browsing as `{user_id}`")
    else:
        st.info("Logged out — showing trending items to everyone.")

# ---------------------------------------------------------------- card renderers
def _filter(items, query):
    if not query:
        return items
    q = query.lower()
    return [it for it in items if q in (it.get("title") or it.get("item_id") or "").lower()]


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
    shown = _filter(items, search_query)
    if not shown:
        st.caption("No results match your search.")
        return
    tint_bg, tint_fg = _category_tint(model_label)
    rows = [shown[i:i + cols_n] for i in range(0, len(shown), cols_n)]
    for row in rows:
        cols = st.columns(cols_n)
        for col, it in zip(cols, row):
            with col:
                title = (it.get("title") or it.get("item_id"))[:70]
                meta = " · ".join(filter(None, [it.get("category"), it.get("brand")]))
                price = f"${it['price']:.2f}" if it.get("price") else ""
                if it.get("image_url"):
                    media = f'<img class="ep-card-img" src="{it["image_url"]}" />'
                else:
                    bg, _ = _category_tint(it.get("category") or "")
                    icon = _category_icon(it.get("category"), title)
                    media = f'<div class="ep-card-placeholder" style="background:{bg};">{icon}</div>'
                card_html = (
                    f'<div class="ep-card">{media}<div class="ep-card-title">{title}</div>'
                    f'<div class="ep-card-meta">{meta}</div><div class="ep-card-price">{price}</div>'
                    f'<span class="ep-card-badge" style="background:{tint_bg};color:{tint_fg};">{model_label}</span>'
                    f'<div class="ep-card-hint">View details →</div></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("View", key=f"{key_prefix}-{it['item_id']}"):
                    _select_item(it["item_id"])


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
                if st.button("View", key=f"{key_prefix}-h-{it['item_id']}"):
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
    it = st.session_state.item_cache.get(item_id, {"item_id": item_id})
    title = it.get("title") or item_id
    meta = " · ".join(filter(None, [it.get("category"), it.get("brand")]))
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
    if st.button("✕ Close", key="close-detail"):
        st.session_state.selected_item = None
        st.rerun()

    st.markdown("**Similar items**")
    sim = get(f"/similar/{item_id}?n=10")
    if sim:
        render_hscroll(sim["items"], key_prefix="sim")


# ---------------------------------------------------------------- main content
if st.session_state.selected_item:
    render_detail_panel()

if not user_id:
    st.markdown('<div class="ep-section-title">Trending electronics</div>', unsafe_allow_html=True)
    data = get("/popular?n=20")
    if data:
        model_caption(data["model_label"], data["latency_ms"])
        render_grid(data["items"], data["model_label"], key_prefix="pop")
else:
    st.markdown('<div class="ep-section-title">For you</div>', unsafe_allow_html=True)
    data = get(f"/recommend/{user_id}?n=20&temperature={temp}")
    if data:
        model_caption(data["model_label"], data["latency_ms"])
        render_grid(data["items"], data["model_label"], key_prefix="rec")

    byl = get(f"/because-you-liked/{user_id}?n=10")
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
