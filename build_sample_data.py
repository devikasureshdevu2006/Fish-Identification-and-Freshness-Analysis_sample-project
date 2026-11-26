import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "sample_dataset")
DEMO_DIR = os.path.join(BASE_DIR, "demo_test_images")

SPECIES_PROFILES = {
    "Karimeen": {"base_hue": 22, "stripe_step": 18, "fresh_sat": 145},
    "Tilapia": {"base_hue": 95, "stripe_step": 32, "fresh_sat": 135},
    "Seer Fish": {"base_hue": 110, "stripe_step": 48, "fresh_sat": 140},
}


def _make_sample_image(
    species: str, is_fresh: bool, seed: int
) -> np.ndarray:
  rng = np.random.default_rng(seed)
  prof = SPECIES_PROFILES[species]

  h, w = 256, 256
  if is_fresh:
    sat_base = prof["fresh_sat"]
    val_base = 205
  else:
    # Non-fresh: dull saturation, low contrast/blur, and bright bloated belly region
    sat_base = 12
    val_base = 55

  hue_plane = np.full((h, w), prof["base_hue"], dtype=np.float32)
  for x in range(0, w, prof["stripe_step"]):
    hue_plane[:, x : x + 4] = (prof["base_hue"] + 25) % 180

  hue_plane += rng.normal(0, 3.0 if is_fresh else 15.0, (h, w))
  hue_u8 = np.clip(hue_plane, 0, 179).astype(np.uint8)

  sat_plane = np.clip(
      sat_base + rng.normal(0, 18.0 if is_fresh else 5.0, (h, w)), 0, 255
  ).astype(np.uint8)
  val_plane = np.clip(
      val_base + rng.normal(0, 25.0 if is_fresh else 5.0, (h, w)), 0, 255
  ).astype(np.uint8)

  if is_fresh:
    for _ in range(25):
      cy = int(rng.integers(25, h - 25))
      cx = int(rng.integers(25, w - 25))
      cv2.circle(val_plane, (cx, cy), int(rng.integers(3, 7)), 250, -1)
    cv2.rectangle(val_plane, (40, 50), (216, 200), 230, 2)
  else:
    # Simulate bloated belly region + blurry/dissolved look
    val_plane[120:230, 50:206] = 235
    sat_plane[120:230, 50:206] = 8
    val_plane = cv2.GaussianBlur(val_plane, (13, 13), 0)
    sat_plane = cv2.GaussianBlur(sat_plane, (13, 13), 0)

  hsv = np.stack([hue_u8, sat_plane, val_plane], axis=-1)
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main() -> None:
  os.makedirs(DEMO_DIR, exist_ok=True)
  total_created = 0

  for species in SPECIES_PROFILES:
    for label in ["Fresh", "Non-Fresh"]:
      out_dir = os.path.join(DATASET_DIR, species, label)
      os.makedirs(out_dir, exist_ok=True)
      is_fresh = label == "Fresh"

      for i in range(30):
        img = _make_sample_image(
            species, is_fresh=is_fresh, seed=total_created + 100
        )
        fname = f"{species.lower().replace(' ', '_')}_{label.lower().replace('-', '_')}_{i:02d}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), img)
        total_created += 1

      demo_img = _make_sample_image(
          species, is_fresh=is_fresh, seed=900 + total_created
      )
      demo_name = f"demo_{species.lower().replace(' ', '_')}_{label.lower().replace('-', '_')}.jpg"
      cv2.imwrite(os.path.join(DEMO_DIR, demo_name), demo_img)

  print(
      f"Created {total_created} sample training images and demo test images"
      " successfully."
  )


if __name__ == "__main__":
  main()