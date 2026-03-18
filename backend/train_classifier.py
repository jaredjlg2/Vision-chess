"""
train_classifier.py
-------------------
Train a HOG + SVM piece classifier from labeled square images.

Usage:
    cd backend
    python train_classifier.py

Prerequisites:
    - pip install scikit-learn scikit-image joblib
    - Collect training data first via POST /api/collect-training-data
      (or place labeled 100x100 PNG images in training_data/<label>/ manually)

Output:
    Saves piece_model.joblib to the backend/ directory.
    The piece_classifier.py will automatically use this model if present.
"""

import os
import sys

import cv2
import joblib
import numpy as np
from skimage.feature import hog as sk_hog
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), "training_data")
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "piece_model.joblib")
SQUARE_SIZE = 100

# HOG parameters — must match piece_classifier.py exactly
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (10, 10)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

# SVM parameters
SVM_C = 10
SVM_GAMMA = "scale"
SVM_KERNEL = "rbf"

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_hog(image_path: str) -> np.ndarray:
    """Load an image and return its HOG feature vector."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (SQUARE_SIZE, SQUARE_SIZE))
    features = sk_hog(
        img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
    )
    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.isdir(TRAINING_DATA_DIR):
        print(
            f"ERROR: Training data directory not found: {TRAINING_DATA_DIR}\n"
            "  Run the app and POST to /api/collect-training-data first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Scanning training data in: {TRAINING_DATA_DIR}")

    X: list[np.ndarray] = []
    y: list[str] = []
    label_counts: dict[str, int] = {}

    for label in sorted(os.listdir(TRAINING_DATA_DIR)):
        label_dir = os.path.join(TRAINING_DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        images = [
            f for f in os.listdir(label_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not images:
            print(f"  [{label}] No images found — skipping.")
            continue

        for filename in images:
            path = os.path.join(label_dir, filename)
            try:
                features = _extract_hog(path)
                X.append(features)
                y.append(label)
            except (IOError, cv2.error, Exception) as exc:
                print(f"  WARNING: Skipping {path}: {exc}", file=sys.stderr)

        label_counts[label] = len(images)
        print(f"  [{label}] {len(images)} images loaded.")

    if len(X) == 0:
        print(
            "ERROR: No training images found.  Collect data first.",
            file=sys.stderr,
        )
        sys.exit(1)

    X_arr = np.array(X)
    y_arr = np.array(y)

    print(f"\nTotal samples: {len(X_arr)}")
    print(f"Feature vector length: {X_arr.shape[1]}")

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_arr
    )
    print(f"Training set: {len(X_train)}  |  Test set: {len(X_test)}")

    # --- Train SVM ---
    print("\nTraining SVM (this may take a moment)…")
    clf = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, probability=True)
    clf.fit(X_train, y_train)
    print("Training complete.")

    # --- Evaluate ---
    y_pred = clf.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # --- Save model ---
    joblib.dump(clf, MODEL_OUTPUT_PATH)
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print("\nDone!  Restart the backend to pick up the new model.")


if __name__ == "__main__":
    main()
