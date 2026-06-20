"""Streamlit frontend. Calls the FastAPI service and renders item cards with
images. Each section uses a DIFFERENT model, just like a real product:

  Homepage (logged out)  -> /popular            (Popularity)
  Homepage (logged in)   -> /recommend          (Two-tower -> LightGBM)
  "Because you liked X"  -> /because-you-liked   (embedding cosine)
  Item page similar      -> /similar             (embedding cosine)
"""
import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://api:8000")
st.set_page_config(page_title="Electronics RecSys", layout="wide")


def card_row(items, label):
    st.caption(f"model: {label}")
    cols = st.columns(min(len(items), 5) or 1)
    for col, it in zip(cols * 5, items):
        with col:
            if it.get("image_url"):
                st.image(it["image_url"], use_container_width=True)
            st.markdown(f"**{(it.get('title') or it['item_id'])[:60]}**")
            meta = " · ".join(filter(None, [it.get("category"), it.get("brand")]))
            if meta:
                st.caption(meta)
            if it.get("price"):
                st.caption(f"${it['price']:.2f}")


def get(path):
    try:
        r = requests.get(f"{API}{path}", timeout=10); r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error on {path}: {e}"); return None


st.sidebar.title("Electronics RecSys")
user_id = st.sidebar.text_input("Log in (user_id)", "")
temp = st.sidebar.slider("Diversity (temperature)", 0.0, 3.0, 1.0, 0.1)
st.sidebar.caption("Recs resample on reload via temperature.")

st.title("Trending Electronics" if not user_id else "For You")

if not user_id:
    data = get("/popular?n=10")
    if data:
        card_row(data["items"], data["model_label"])
        st.caption(f"latency: {data['latency_ms']}")
else:
    data = get(f"/recommend/{user_id}?n=10&temperature={temp}")
    if data:
        card_row(data["items"], data["model_label"])
        st.caption(f"latency: {data['latency_ms']}")
    st.subheader("Because you liked…")
    byl = get(f"/because-you-liked/{user_id}?n=10")
    if byl:
        card_row(byl["items"], byl["model_label"])

st.divider()
st.subheader("Item page demo — similar items")
item_id = st.text_input("item_id", "")
if item_id:
    sim = get(f"/similar/{item_id}?n=10")
    if sim:
        card_row(sim["items"], sim["model_label"])
