from pathlib import Path
import cv2
import joblib
import numpy as np
import streamlit as st

from freshness_features import predict_freshness

THRESHOLD = 70.0
MODEL_PATH = Path(__file__).resolve().parent / "freshness_model.joblib"

st.set_page_config(
    page_title="Fish Freshness Detector", page_icon="🐟", layout="centered"
)
st.title("🐟 Fish Freshness Detector")
st.write(
    f"Cutoff: **{THRESHOLD:.0f}%** — **{THRESHOLD:.0f}%** or above is"
    " **Fresh**, below is **Non-Fresh**."
)


@st.cache_resource
def load_model():
  if MODEL_PATH.exists():
    return joblib.load(MODEL_PATH)
  return None


model = load_model()
uploaded = st.file_uploader(
    "Upload a fish image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:
  file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
  bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  if bgr_img is None:
    st.error("Could not read the image. Please choose another photo.")
  else:
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    st.image(rgb_img, caption="Uploaded Fish Image", use_container_width=True)

    result = predict_freshness(bgr_img, model=model, threshold=THRESHOLD)
    fresh_pct = result["fresh_pct"]
    status = result["status"]

    st.divider()
    if status == "Fresh":
      st.success(f"### ✅ Status: FRESH ({fresh_pct}%)")
    else:
      st.error(f"### ❌ Status: NON-FRESH ({fresh_pct}%)")

    st.write("**Symptom Breakdown (0–100):**")
    for k, v in result["symptoms"].items():
      st.write(f"- **{k}:** {v}%")