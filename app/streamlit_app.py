import json
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")
st.title("🖼️ AI-Powered Image Classification")
st.caption("CIFAR-10 • Transfer Learning • ResNet18 / MobileNetV3")

API = "https://ai-image-classification-platform.onrender.com"
uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    st.image(uploaded, caption=uploaded.name, width="stretch")
    if st.button("Classify image", type="primary"):
        with st.spinner("Predicting..."):
            resp = requests.post(f"{API}/predict", files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}, timeout=60)
        if resp.ok:
            data = resp.json()
            st.success(f"Prediction: {data['prediction']['class']} ({data['prediction']['confidence']:.1%})")
            rows = [{"Class": x["class"], "Confidence": f"{x['confidence']:.1%}"} for x in data["top_5"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.error(resp.text)

st.divider()
st.subheader("Prediction history")
try:
    h = requests.get(f"{API}/history?limit=20", timeout=5)
    if h.ok and h.json()["items"]:
        rows = h.json()["items"]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No prediction history yet.")
except Exception:
    st.info("Start the FastAPI server to view history.")
