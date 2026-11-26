import cv2
import numpy as np


def _clamp_0_100(val: float) -> float:
    return float(np.clip(val, 0.0, 100.0))


def _norm_range(val: float, low: float, high: float) -> float:
    if high <= low:
        return 50.0
    return _clamp_0_100(((val - low) / (high - low)) * 100.0)


def compute_built_in_freshness(bgr_img: np.ndarray) -> dict:
    """
    Calibrated OpenCV heuristic for app.py:
    - Prevents white ice / bright background & glare from collapsing Surface score to ~5%.
    - Avoids harsh penalties when Gill or Belly is partially submerged/obscured on ice.
    - Preserves sensitivity to clear eye detail, healthy visual structure, and real spoilage/damage.
    """
    if bgr_img is None or bgr_img.size == 0:
        raise ValueError("Invalid image provided to compute_built_in_freshness")

    img = cv2.resize(bgr_img, (224, 224), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)

    val_mean = float(np.mean(hsv[:, :, 2]))
    val_std = float(np.std(hsv[:, :, 2]))
    lap_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
    edges = cv2.Canny(gray_u8, 50, 150)
    edge_density = float(np.mean(edges > 0)) * 100.0

    # 1. Surface Freshness: reward sharpness, edge definition, and natural tonal variation;
    # only penalize extreme blown-out glare (>252) or heavy dark decay (<40).
    bright_glare_ratio = float(np.mean((hsv[:, :, 2] > 252) & (hsv[:, :, 1] < 20)))
    dark_rot_ratio = float(np.mean(gray < 40))

    surface_score = _clamp_0_100(
        0.35 * _norm_range(lap_var, 25.0, 350.0)
        + 0.25 * _norm_range(edge_density, 1.0, 12.0)
        + 0.20 * _norm_range(val_std, 12.0, 65.0)
        + 0.20 * _norm_range(val_mean, 60.0, 245.0)
        - bright_glare_ratio * 25.0
        - dark_rot_ratio * 70.0
    )

    # 2. Eye Condition: sharpness & contrast across main subject region
    eye_score = _clamp_0_100(
        0.50 * _norm_range(lap_var, 30.0, 320.0)
        + 0.30 * _norm_range(val_std, 14.0, 60.0)
        + 0.20 * _norm_range(edge_density, 1.2, 14.0)
    )

    # 3. Gill Color / Condition: if gill/redness/edge cues are limited or obscured on ice,
    # keep a balanced baseline (~72) instead of dropping sharply.
    b_ch = img[:, :, 0].astype(np.float32)
    g_ch = img[:, :, 1].astype(np.float32)
    r_ch = img[:, :, 2].astype(np.float32)

    red_cues = float(np.mean((r_ch > g_ch + 12.0) & (r_ch > b_ch + 12.0) & (r_ch > 65.0)))
    dark_sludge = float(np.mean((r_ch < 55.0) & (g_ch < 55.0) & (b_ch < 55.0)))

    if red_cues < 0.01 and edge_density < 2.0:
        gill_score = 72.0
    else:
        gill_score = _clamp_0_100(
            68.0
            + _norm_range(red_cues, 0.01, 0.12) * 0.25
            + _norm_range(edge_density, 1.5, 14.0) * 0.15
            - dark_sludge * 80.0
        )

    # 4. Belly Swelling Check: avoid punishing ice-submerged or uniform bright bellies
    belly_roi = gray[115:215, 35:190]
    belly_std = float(np.std(belly_roi))
    belly_dark = float(np.mean(belly_roi < 45))

    if belly_std < 10.0:
        belly_score = 74.0
    else:
        excess_std_penalty = max(0.0, (belly_std - 65.0) * 0.5)
        belly_score = _clamp_0_100(84.0 - (belly_dark * 90.0) - excess_std_penalty)

    freshness_pct = round(
        _clamp_0_100(
            0.25 * surface_score
            + 0.30 * eye_score
            + 0.25 * gill_score
            + 0.20 * belly_score
        ),
        1,
    )

    return {
        "freshness_pct": freshness_pct,
        "gill_score": round(gill_score, 1),
        "eye_score": round(eye_score, 1),
        "belly_score": round(belly_score, 1),
        "surface_score": round(surface_score, 1),
    }