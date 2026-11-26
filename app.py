import os
import json
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

try:
    from freshness_features import compute_symptom_scores as _external_symptom_scores
except Exception:
    _external_symptom_scores = None

st.set_page_config(page_title="Fish Identification & Freshness Analysis", layout="wide")


def analyze_fish_freshness_opencv(img_bgr: np.ndarray) -> dict:
    """
    Conservative OpenCV freshness estimator targeting gill redness/clarity,
    eye clarity/contrast, belly/flesh integrity, and surface sheen without false glare boost.
    """
    h, w, _ = img_bgr.shape
    resized = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # 1. Surface & Glare check: high brightness/glare on wet/stale skin reduces true freshness score
    glare_ratio = float(np.mean(val > 215.0))
    dark_rot_ratio = float(np.mean(val < 45.0))
    mid_sat = float(np.mean(np.clip((sat / 110.0) * 75.0, 0.0, 80.0)))
    surface_score = mid_sat - (glare_ratio * 120.0) - (dark_rot_ratio * 110.0)
    surface_score = float(np.clip(surface_score, 5.0, 85.0))

    # 2. Gill region proxy (upper-middle / anterior section): bright red hue in HSV (0-12 or 168-180 with decent sat)
    gill_crop = resized[int(0.15 * 512):int(0.75 * 512), int(0.15 * 512):int(0.85 * 512)]
    g_hsv = cv2.cvtColor(gill_crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    red_mask = (((g_hsv[:, :, 0] <= 12.0) | (g_hsv[:, :, 0] >= 168.0)) & (g_hsv[:, :, 1] > 75.0) & (g_hsv[:, :, 2] > 55.0))
    red_ratio = float(np.mean(red_mask))
    gill_score = float(np.clip(15.0 + (red_ratio * 320.0) - (dark_rot_ratio * 90.0), 5.0, 90.0))
    gill_visibility = "Detected" if red_ratio > 0.03 else "Dull / Limited"

    # 3. Eye clarity proxy: local contrast & dark pupil vs cloudy/dull iris
    eye_crop = gray[int(0.55 * 512):int(0.95 * 512), int(0.40 * 512):int(0.95 * 512)]
    lap_var = float(cv2.Laplacian(eye_crop, cv2.CV_32F).var())
    eye_contrast = float(np.std(eye_crop))
    eye_score = float(np.clip(12.0 + (min(lap_var, 400.0) / 400.0) * 35.0 + (min(eye_contrast, 70.0) / 70.0) * 25.0, 5.0, 85.0))

    # 4. Belly & Flesh integrity proxy: penalize strong edges/cuts/tears/cavities inside body region
    body_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0).astype(np.uint8), 60, 150)
    edge_density = float(np.mean(body_edges > 0))
    # High edge density from cuts/open cavities/tears lowers integrity score
    belly_score = float(np.clip(78.0 - (edge_density * 420.0) - (dark_rot_ratio * 100.0), 5.0, 85.0))
    belly_visibility = "Cut / Irregular" if edge_density > 0.08 else "Intact / Check"

    # Weighted composite favoring conservative freshness estimation
    composite = (
        0.30 * gill_score
        + 0.25 * eye_score
        + 0.25 * belly_score
        + 0.20 * surface_score
    )

    # Extra penalty if image shows heavy cutting/damage or very low overall saturation
    if edge_density > 0.09 or float(np.mean(sat)) < 55.0:
        composite -= 12.0

    composite = round(float(np.clip(composite, 0.0, 95.0)), 1)

    return {
        "gill_score": round(gill_score, 1),
        "gill_visibility": gill_visibility,
        "eye_score": round(eye_score, 1),
        "belly_score": round(belly_score, 1),
        "belly_visibility": belly_visibility,
        "surface_score": round(surface_score, 1),
        "symptom_composite": composite,
    }


def get_freshness_scores(img_bgr: np.ndarray, use_external_module: bool) -> dict:
    if use_external_module and _external_symptom_scores is not None:
        try:
            res = _external_symptom_scores(img_bgr)
            if isinstance(res, dict) and "symptom_composite" in res:
                return res
        except Exception:
            pass
    return analyze_fish_freshness_opencv(img_bgr)


def build_cnn(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


@st.cache_resource
def load_checkpoint(path: str):
    if not path or not os.path.exists(path):
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = build_cnn(len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]


def predict_image(model, classes, img_rgb: np.ndarray):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tfm(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx] * 100.0), dict(zip(classes, [round(float(p * 100), 2) for p in probs]))


st.title("Fish Identification & Freshness Analysis")
st.caption("Stack: Python, PyTorch/Torchvision (Deep Learning), OpenCV (cv2), NumPy, Pandas, Scikit-Learn & Streamlit")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Freshness Threshold (%)", 50, 95, 70, 1)
    use_external = st.checkbox("Use freshness_features.py if available", value=False)
    species_ckpt = st.text_input("Species Checkpoint", "models/species_model.pth")
    freshness_ckpt = st.text_input("Freshness Checkpoint", "models/freshness_model.pth")
    show_debug = st.checkbox("Show Score Breakdown", value=True)

tab1, tab2 = st.tabs(["Inspection Dashboard", "Evaluation Metrics (70/15/15)"])

if "logs" not in st.session_state:
    st.session_state["logs"] = []

with tab1:
    source = st.radio("Input Source", ["Upload Image", "Camera Capture"], horizontal=True)
    file = (
        st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"])
        if source == "Upload Image"
        else st.camera_input("Capture image")
    )

    if file is not None:
        file_bytes = np.asarray(bytearray(file.getvalue()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Unable to decode image. Please upload a valid JPG or PNG file.")
            st.stop()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        symptoms = get_freshness_scores(img_bgr, use_external)

        sp_model, sp_classes = load_checkpoint(species_ckpt)
        fr_model, fr_classes = load_checkpoint(freshness_ckpt)

        if sp_model is not None:
            species_name, species_conf, _ = predict_image(sp_model, sp_classes, img_rgb)
        else:
            species_name, species_conf = "Species Preview (Train species_model.pth)", 0.0

        fresh_prob = None
        if fr_model is not None:
            _, _, fr_probs = predict_image(fr_model, fr_classes, img_rgb)
            fresh_prob = next(
                (v for k, v in fr_probs.items() if "fresh" in k.lower() and "non" not in k.lower() and "not" not in k.lower()),
                max(fr_probs.values()),
            )
            freshness_pct = round(0.70 * fresh_prob + 0.30 * float(symptoms["symptom_composite"]), 1)
        else:
            freshness_pct = round(float(symptoms["symptom_composite"]), 1)

        decision = "FRESH" if freshness_pct >= threshold else "NON-FRESH"

        c_left, c_right = st.columns([1, 1.3])
        with c_left:
            st.image(img_rgb, caption="Input Image (224x224)", use_container_width=True)
        with c_right:
            k1, k2, k3 = st.columns(3)
            k1.metric("Species", species_name.split(" (")[0], f"{species_conf:.1f}%" if species_conf > 0 else "Preview")
            k2.metric("Freshness Score", f"{freshness_pct:.1f}%", f"Threshold: {threshold}%")
            k3.metric("Status", decision)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Gill Color / Condition", f"{symptoms['gill_score']}/100", symptoms.get("gill_visibility", "Visible"))
            s2.metric("Eye Condition", f"{symptoms['eye_score']}/100", "Visible")
            s3.metric("Belly Swelling Check", f"{symptoms['belly_score']}/100", symptoms.get("belly_visibility", "Visible"))
            s4.metric("Surface Freshness", f"{symptoms['surface_score']}/100", "Visible")

            if show_debug:
                st.caption(
                    f"Analyzer: {'freshness_features.py' if (use_external and _external_symptom_scores) else 'Built-in OpenCV Heuristic'} | "
                    f"Model Fresh Prob: {f'{fresh_prob:.1f}%' if fresh_prob is not None else 'Not Loaded'} | "
                    f"Symptom Composite: {symptoms['symptom_composite']}%"
                )

        row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Image": getattr(file, "name", "camera.jpg"),
            "Species": species_name,
            "Freshness_%": freshness_pct,
            "Threshold_%": threshold,
            "Status": decision,
            "Gill_Score": symptoms["gill_score"],
            "Eye_Score": symptoms["eye_score"],
            "Belly_Score": symptoms["belly_score"],
            "Surface_Score": symptoms["surface_score"],
        }
        if (
            not st.session_state["logs"]
            or st.session_state["logs"][-1]["Image"] != row["Image"]
            or st.session_state["logs"][-1]["Status"] != row["Status"]
        ):
            st.session_state["logs"].append(row)

    if st.session_state["logs"]:
        df_logs = pd.DataFrame(st.session_state["logs"])
        st.dataframe(df_logs, use_container_width=True)
        st.download_button(
            "Download CSV Report",
            df_logs.to_csv(index=False).encode("utf-8"),
            "fish_freshness_report.csv",
            "text/csv",
        )

with tab2:
    for task_key, title in [("species", "Species Identification"), ("freshness", "Freshness Classification")]:
        m_path = f"models/{task_key}_metrics.json"
        st.subheader(title)
        if os.path.exists(m_path):
            with open(m_path) as f:
                m = json.load(f)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{m['accuracy']}%")
            m2.metric("Precision", f"{m['precision']}%")
            m3.metric("Recall", f"{m['recall']}%")
            m4.metric("F1-Score", f"{m['f1_score']}%")
            st.dataframe(pd.DataFrame(m["confusion_matrix"], index=m["classes"], columns=m["classes"]))
        else:
            st.info(f"Run `python train.py --task {task_key} --data_dir <path>` to generate `{m_path}`.")