"""
Giao diện web đơn giản (Streamlit) — upload ảnh → phân loại dog/cat.
Chạy: streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.predict import load_artifact, predict_image


st.set_page_config(page_title="Phân loại động vật", page_icon="🐾", layout="centered")

st.title("Phân loại động vật (LBP + Random Forest)")
st.caption(
    "224×224 · CLAHE · Đa vùng LBP (toàn ảnh + 4 góc + giữa) · P=8,R=1 · RF balanced"
)

if not config.MODEL_PATH.is_file():
    st.error(
        f"Chưa có mô hình tại `{config.MODEL_PATH}`.\n\n"
        "1. Đặt ảnh vào `data/raw/<tên_lớp>/`\n"
        "2. Chạy: `python -m src.train`"
    )
    st.stop()

try:
    artifact = load_artifact()
    class_names = artifact["class_names"]
except Exception as e:
    st.error(f"Không tải được mô hình: {e}")
    st.stop()

st.info(f"Các lớp hiện tại: **{', '.join(class_names)}**")

uploaded = st.file_uploader(
    "Chọn ảnh (JPG/PNG)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded is not None:
    st.image(uploaded, caption="Ảnh đầu vào", use_container_width=True)

    suffix = Path(uploaded.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    try:
        label, probs = predict_image(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    st.subheader(f"Kết quả: **{label}**")

    rows = sorted(probs.items(), key=lambda x: -x[1])
    for name, p in rows:
        st.write(f"**{name}**: {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))

with st.expander("Cấu hình mô hình"):
    cfg = artifact.get("config", {})
    st.json(cfg)
