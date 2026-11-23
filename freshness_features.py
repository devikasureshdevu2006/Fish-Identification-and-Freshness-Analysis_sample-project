import cv2
import numpy as np


def _clip_100(val: float) -> float:
  return float(np.clip(val, 0.0, 100.0))


def _scale(val: float, low: float, high: float) -> float:
  if high <= low:
    return 50.0
  return _clip_100(((val - low) / (high - low)) * 100.0)


def extract_features(bgr_img: np.ndarray) -> np.ndarray:
  """Extracts 8 clear visual features from a BGR fish image."""
  if bgr_img is None or bgr_img.size == 0:
    raise ValueError("Invalid image provided to extract_features")

  img = cv2.resize(bgr_img, (256, 256), interpolation=cv2.INTER_AREA)
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
  gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = gray_u8.astype(np.float32)

  sat_mean = float(np.mean(hsv[:, :, 1]))
  sat_std = float(np.std(hsv[:, :, 1]))
  val_mean = float(np.mean(hsv[:, :, 2]))
  val_std = float(np.std(hsv[:, :, 2]))

  lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
  lap_var = float(np.var(lap))
  edges = cv2.Canny(gray_u8, 60, 150)
  edge_ratio = float(np.mean(edges > 0)) * 100.0

  belly = gray[125:235, 45:211]
  belly_mean = float(np.mean(belly))
  belly_std = float(np.std(belly))

  return np.array(
      [
          sat_mean,
          sat_std,
          val_mean,
          val_std,
          lap_var,
          edge_ratio,
          belly_mean,
          belly_std,
      ],
      dtype=np.float32,
  )


def compute_symptom_breakdown(bgr_img: np.ndarray) -> dict[str, float]:
  """Returns 0-100 scores for 4 visual checks."""
  feats = extract_features(bgr_img)
  sat_mean, sat_std, val_mean, val_std, lap_var, edge_ratio, belly_mean, belly_std = (
      feats
  )

  # Silver/fresh fish can have moderate saturation + good brightness & sharpness
  gill_score = _clip_100(
      0.35 * _scale(sat_mean, 8.0, 55.0)
      + 0.35 * _scale(val_mean, 70.0, 210.0)
      + 0.30 * _scale(edge_ratio, 2.5, 16.0)
  )

  eye_score = _clip_100(
      0.50 * _scale(lap_var, 50.0, 650.0)
      + 0.30 * _scale(edge_ratio, 2.5, 16.0)
      + 0.20 * _scale(val_std, 18.0, 75.0)
  )

  over_bright_belly = _scale(belly_mean, 210.0, 252.0)
  flat_belly = 100.0 - _scale(belly_std, 6.0, 40.0)
  belly_penalty = 0.55 * over_bright_belly + 0.45 * flat_belly
  belly_score = _clip_100(100.0 - 0.50 * belly_penalty)

  surface_score = _clip_100(
      0.30 * _scale(sat_mean, 8.0, 55.0)
      + 0.40 * _scale(val_mean, 70.0, 210.0)
      + 0.30 * _scale(val_std, 18.0, 75.0)
  )

  return {
      "Gill Condition": round(gill_score, 2),
      "Eye Condition": round(eye_score, 2),
      "Belly Swelling Check": round(belly_score, 2),
      "Surface Freshness": round(surface_score, 2),
  }


def predict_freshness(
    bgr_img: np.ndarray,
    model=None,
    threshold: float = 70.0,
) -> dict:
  symptoms = compute_symptom_breakdown(bgr_img)
  weights = {
      "Gill Condition": 0.30,
      "Eye Condition": 0.25,
      "Belly Swelling Check": 0.25,
      "Surface Freshness": 0.20,
  }
  visual_pct = sum(symptoms[k] * weights[k] for k in symptoms)

  # Give higher weight to visual symptom score so fresh silver/iced fish aren't pulled down
  if model is not None:
    X = extract_features(bgr_img).reshape(1, -1)
    probs = model.predict_proba(X)[0]
    classes = list(model.classes_)
    fresh_idx = classes.index(1) if 1 in classes else 1
    ml_pct = float(probs[fresh_idx]) * 100.0
    fresh_pct = 0.25 * ml_pct + 0.75 * visual_pct
  else:
    fresh_pct = visual_pct

  fresh_pct = round(_clip_100(fresh_pct), 2)
  non_fresh_pct = round(100.0 - fresh_pct, 2)
  status = "Fresh" if fresh_pct >= threshold else "Non-Fresh"

  return {
      "status": status,
      "fresh_pct": fresh_pct,
      "non_fresh_pct": non_fresh_pct,
      "symptoms": symptoms,
  }  