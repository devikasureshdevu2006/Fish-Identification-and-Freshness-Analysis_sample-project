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
import sklearn

try:
    from freshness_features import compute_symptom_scores as _external_symptom_scores
except Exception:
    _external_symptom_scores = None

st.set_page_config(page_title="Fish Identification & Freshness Analysis", layout="wide")


def _clamp_0_100(val: float) -> float:
    return float(np.clip(val, 0.0, 100.0))


def _norm_range(val: float, low: float, high: float) -> float:
    if high <= low:
        return 50.0
    return _clamp_0_100(((val - low) / (high - low)) * 100.0)


def analyze_fish_freshness_opencv(img_bgr: np.ndarray) -> dict:
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Invalid image provided to analyze_fish_freshness_opencv")

    img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)

    b_ch = img[:, :, 0].astype(np.float32)
    g_ch = img[:, :, 1].astype(np.float32)
    r_ch = img[:, :, 2].astype(np.float32)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Ignore white ice background so ice doesn't lower score for fresh fish
    ice_bg_mask = (val > 185.0) & (sat < 35.0)
    fg_mask = ~ice_bg_mask
    fg_ratio = float(np.mean(fg_mask))

    if fg_ratio > 0.12:
        fg_sat_mean = float(np.mean(sat[fg_mask]))
        fg_val_std = float(np.std(val[fg_mask]))
        # Cool metallic blue/silver tone common in fresh marine fish
        cool_fresh_ratio = float(np.mean((b_ch[fg_mask] >= r_ch[fg_mask] - 4.0) & (val[fg_mask] > 75.0)))
        # Warm pinkish/brownish/yellowish discoloration common in stale/cut fish
        warm_stale_ratio = float(np.mean((r_ch[fg_mask] > b_ch[fg_mask] + 18.0) & (sat[fg_mask] > 25.0)))
        dark_rot_ratio = float(np.mean(gray[fg_mask] < 50.0))
    else:
        fg_sat_mean = float(np.mean(sat))
        fg_val_std = float(np.std(val))
        cool_fresh_ratio = float(np.mean((b_ch >= r_ch - 4.0) & (val > 75.0)))
        warm_stale_ratio = float(np.mean((r_ch > b_ch + 18.0) & (sat > 25.0)))
        dark_rot_ratio = float(np.mean(gray < 50.0))

    lap_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())

    # 1. Surface Freshness (rewards cool metallic sheen; penalizes warm brownish/pinkish spoilage)
    surface_score = _clamp_0_100(
        46.0
        + 0.18 * _norm_range(lap_var, 45.0, 420.0)
        + 0.18 * _norm_range(fg_val_std, 18.0, 68.0)
        + cool_fresh_ratio * 24.0
        - warm_stale_ratio * 34.0
        - dark_rot_ratio * 90.0
    )

    # 2. Eye Condition (clear dark pupil + bright ring = high local contrast & dark-to-bright spread)
    eye_crop = gray_u8[35:155, 50:200]
    eye_lap = float(cv2.Laplacian(eye_crop, cv2.CV_64F).var()) if eye_crop.size > 0 else lap_var
    eye_con = float(np.std(eye_crop)) if eye_crop.size > 0 else fg_val_std
    eye_score = _clamp_0_100(
        44.0
        + 0.28 * _norm_range(eye_lap, 45.0, 450.0)
        + 0.28 * _norm_range(eye_con, 20.0, 68.0)
        + cool_fresh_ratio * 10.0
        - warm_stale_ratio * 22.0
    )

    # 3. Gill Color / Condition
    bright_red_mask = (r_ch > g_ch + 38.0) & (r_ch > b_ch + 38.0) & (r_ch > 130.0)
    bright_red_ratio = float(np.mean(bright_red_mask))
    pale_brown_gill = float(np.mean((r_ch > g_ch + 10.0) & (r_ch <= g_ch + 38.0) & (r_ch < 155.0)))
    dark_sludge = float(np.mean((r_ch < 65.0) & (g_ch < 65.0) & (b_ch < 65.0)))

    if bright_red_ratio >= 0.015:
        gill_score = _clamp_0_100(
            74.0
            + min(bright_red_ratio * 260.0, 18.0)
            - dark_sludge * 100.0
            - pale_brown_gill * 45.0
        )
    else:
        gill_score = _clamp_0_100(
            72.0
            + cool_fresh_ratio * 10.0
            - dark_sludge * 110.0
            - pale_brown_gill * 42.0
        )

    # 4. Belly Swelling Check
    belly_roi = gray[115:215, 35:190]
    belly_std = float(np.std(belly_roi))
    belly_dark = float(np.mean(belly_roi < 50.0))
    belly_score = _clamp_0_100(
        74.0
        + cool_fresh_ratio * 8.0
        - (belly_dark * 110.0)
        - warm_stale_ratio * 24.0
        - max(0.0, (belly_std - 62.0) * 0.5)
    )

    composite = round(
        _clamp_0_100(
            0.30 * surface_score
            + 0.25 * eye_score
            + 0.25 * gill_score
            + 0.20 * belly_score
        ),
        1,
    )

    return {
        "gill_score": round(gill_score, 1),
        "gill_visibility": "Detected" if bright_red_ratio >= 0.015 else "Closed / Check",
        "eye_score": round(eye_score, 1),
        "belly_score": round(belly_score, 1),
        "belly_visibility": "Intact / Normal" if belly_dark < 0.08 else "Check",
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
    inspector_email = st.text_input("Inspector Email", value="")
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

        clean_email = inspector_email.strip() if inspector_email.strip() else "Not Provided"
        row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Email": clean_email,
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
            or st.session_state["logs"][-1]["Email"] != row["Email"]
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