from pathlib import Path
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from freshness_features import extract_features, predict_freshness

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "freshness_model.joblib"


def _synthetic_fresh_image(seed: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  h, w = 256, 256
  hue = np.full((h, w), int(rng.integers(15, 130)), dtype=np.float32)
  for x in range(0, w, 20):
    hue[:, x : x + 5] = (hue[:, x : x + 5] + 25) % 180
  hue_u8 = np.clip(hue + rng.normal(0, 2.0, (h, w)), 0, 179).astype(np.uint8)

  sat_u8 = np.clip(
      rng.normal(135, 18, (h, w)), 85, 230
  ).astype(np.uint8)
  val_u8 = np.clip(
      rng.normal(185, 25, (h, w)), 110, 245
  ).astype(np.uint8)

  # Crisp edges & sheen highlights
  for _ in range(30):
    cx = int(rng.integers(20, w - 20))
    cy = int(rng.integers(20, h - 20))
    cv2.circle(val_u8, (cx, cy), int(rng.integers(3, 8)), 245, -1)
  cv2.line(val_u8, (30, 80), (225, 80), 230, 2)
  cv2.line(val_u8, (30, 160), (225, 160), 220, 2)

  hsv = np.stack([hue_u8, sat_u8, val_u8], axis=-1)
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _synthetic_non_fresh_image(seed: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  h, w = 256, 256
  hue_u8 = np.clip(rng.normal(25, 18, (h, w)), 0, 179).astype(np.uint8)
  # Dull/pale saturation & low/uneven contrast
  sat_u8 = np.clip(rng.normal(14, 6, (h, w)), 0, 35).astype(np.uint8)
  val_u8 = np.clip(rng.normal(75, 12, (h, w)), 35, 120).astype(np.uint8)

  # Bloated / washed-out belly patch
  val_u8[125:235, 45:211] = 225
  sat_u8[125:235, 45:211] = 5

  # Blurry / dissolved appearance
  val_u8 = cv2.GaussianBlur(val_u8, (15, 15), 0)
  sat_u8 = cv2.GaussianBlur(sat_u8, (15, 15), 0)

  hsv = np.stack([hue_u8, sat_u8, val_u8], axis=-1)
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def load_dataset():
  X, y = [], []
  fresh_dir = DATA_DIR / "Fresh"
  non_fresh_dir = DATA_DIR / "Non-Fresh"

  exts = {".jpg", ".jpeg", ".png", ".webp"}
  for folder, label in [(fresh_dir, 1), (non_fresh_dir, 0)]:
    if folder.exists():
      for p in folder.rglob("*"):
        if p.suffix.lower() in exts:
          img = cv2.imread(str(p))
          if img is not None:
            X.append(extract_features(img))
            y.append(label)

  # If user hasn't placed enough real images yet, supplement with clear examples
  fresh_count = sum(1 for v in y if v == 1)
  non_fresh_count = sum(1 for v in y if v == 0)

  if fresh_count < 25:
    for i in range(25 - fresh_count):
      X.append(extract_features(_synthetic_fresh_image(1000 + i)))
      y.append(1)

  if non_fresh_count < 25:
    for i in range(25 - non_fresh_count):
      X.append(extract_features(_synthetic_non_fresh_image(2000 + i)))
      y.append(0)

  return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
  X, y = load_dataset()
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  clf = Pipeline([
      ("scaler", StandardScaler()),
      (
          "rf",
          RandomForestClassifier(
              n_estimators=150, random_state=42, class_weight="balanced"
          ),
      ),
  ])
  clf.fit(X_train, y_train)

  preds = clf.predict(X_test)
  acc = accuracy_score(y_test, preds) * 100.0
  joblib.dump(clf, MODEL_PATH)

  print(f"Model saved to: {MODEL_PATH}")
  print(f"Test Accuracy: {acc:.2f}%")

  # Quick sanity check on one Fresh and one Non-Fresh sample
  fresh_res = predict_freshness(_synthetic_fresh_image(777), model=clf, threshold=70.0)
  non_fresh_res = predict_freshness(
      _synthetic_non_fresh_image(888), model=clf, threshold=70.0
  )
  print("Sample Fresh ->", fresh_res["status"], fresh_res["fresh_pct"])
  print("Sample Non-Fresh ->", non_fresh_res["status"], non_fresh_res["fresh_pct"])


if __name__ == "__main__":
  main()