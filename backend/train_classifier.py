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
import warnings

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional ML dependencies (joblib / scikit-image / scikit-learn)
# ---------------------------------------------------------------------------

try:
    import joblib
    from skimage.feature import hog as sk_hog
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    _ML_AVAILABLE = True
except ImportError:
    warnings.warn(
        "scikit-image, joblib, or scikit-learn not installed — "
        "the /api/train endpoint will be unavailable.  "
        "Run: pip install scikit-image joblib scikit-learn",
        stacklevel=1,
    )
    _ML_AVAILABLE = False

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
# Training logic (importable)
# ---------------------------------------------------------------------------

def train_model() -> dict:
    """
    Train the HOG+SVM classifier from labeled images in ``training_data/``.

    Returns
    -------
    dict
        ``{"samples": int, "accuracy": float, "report": str, "label_counts": dict[str, int]}``

    Raises
    ------
    ImportError
        If scikit-image, joblib, or scikit-learn are not installed.
    FileNotFoundError
        If the training data directory does not exist.
    ValueError
        If no training images are found.
    """
    if not _ML_AVAILABLE:
        raise ImportError(
            "scikit-image, joblib, and scikit-learn are required for training.  "
            "Run: pip install scikit-image joblib scikit-learn"
        )

    if not os.path.isdir(TRAINING_DATA_DIR):
        raise FileNotFoundError(
            f"Training data directory not found: {TRAINING_DATA_DIR}\n"
            "  Run the app and POST to /api/collect-training-data first."
        )

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
            continue

        for filename in images:
            path = os.path.join(label_dir, filename)
            try:
                features = _extract_hog(path)
                X.append(features)
                y.append(label)
            except (IOError, cv2.error, Exception):
                pass

        label_counts[label] = len(images)

    if len(X) == 0:
        raise ValueError(
            "No training images found.  Collect data first via "
            "POST /api/collect-training-data."
        )

    X_arr = np.array(X)
    y_arr = np.array(y)

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_arr
    )

    # --- Train SVM ---
    clf = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, probability=True)
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)

    correct = int(np.sum(y_pred == y_test))
    accuracy = correct / len(y_test) if len(y_test) > 0 else 0.0

    # --- Save model ---
    joblib.dump(clf, MODEL_OUTPUT_PATH)

    return {
        "samples": int(len(X_arr)),
        "accuracy": round(accuracy, 4),
        "report": report,
        "label_counts": label_counts,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Scanning training data in: {TRAINING_DATA_DIR}")
    try:
        result = train_model()
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal samples: {result['samples']}")
    for label, count in sorted(result["label_counts"].items()):
        print(f"  [{label}] {count} images loaded.")
    print(f"\nAccuracy: {result['accuracy']:.4f}")
    print("\nClassification report:")
    print(result["report"])
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print("\nDone!  The backend will pick up the new model on the next inference call.")


if __name__ == "__main__":
    main()
