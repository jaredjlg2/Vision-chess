"""
piece_classifier.py
-------------------
Classifies chess pieces in each square of a perspective-corrected 800×800
board image.

Primary path  (when piece_model.joblib is present):
  HOG feature extraction → trained SVM classifier loaded via joblib.

Fallback path (when no model file exists):
  Classical OpenCV heuristics (edge density, blob aspect ratio, circularity,
  diagonal gradients).

Pipeline per square:
  1. Determine empty vs. occupied  — via pixel variance + edge density.
  2. If occupied and model available → HOG + SVM inference.
  3. If no model → heuristic colour + piece-type classification.
"""

import logging
import os
import warnings

import cv2
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Case-safe label helpers (inference-time reverse mapping)
# ---------------------------------------------------------------------------
# Note: collect_training_data._fen_from_safe_label serves the same purpose
# but returns the string "empty" instead of None, because it is used for
# data-recovery tooling.  Here we return None for "empty" to match the
# convention in classify_pieces() where None means an unoccupied square.


def _fen_from_safe_label(safe: str) -> Optional[str]:
    """Convert a safe folder-name label back to a FEN character.

    The trained model predicts safe labels like ``"white_P"``, ``"black_r"``,
    or ``"empty"``.  This function maps them back to the FEN character expected
    by ``classify_pieces``:

    * ``"white_P"`` → ``"P"``
    * ``"black_r"`` → ``"r"``
    * ``"empty"``   → ``None``  (empty square — no FEN character)
    """
    if safe == "empty":
        return None
    if safe.startswith("white_"):
        return safe[len("white_"):]
    if safe.startswith("black_"):
        return safe[len("black_"):]
    # Legacy single-char label (training data collected before this fix):
    # return as-is so old models continue to work until data is re-collected.
    return safe if safe else None

# ---------------------------------------------------------------------------
# Optional ML dependencies (skimage / joblib)
# ---------------------------------------------------------------------------

try:
    import joblib
    from skimage.feature import hog as sk_hog
    _ML_AVAILABLE = True
except ImportError:
    warnings.warn(
        "scikit-image or joblib not installed — falling back to heuristic "
        "piece classifier.  Run: pip install scikit-image joblib",
        stacklevel=1,
    )
    _ML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Model loading (lazy)
# ---------------------------------------------------------------------------

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "piece_model.joblib")
_model = None  # loaded lazily on first inference call


def _load_model():
    """Load the SVM model from disk on first call; return None if unavailable."""
    global _model
    if not _ML_AVAILABLE:
        return None
    if _model is None and os.path.exists(_MODEL_PATH):
        try:
            _model = joblib.load(_MODEL_PATH)
        except Exception as exc:
            warnings.warn(f"Failed to load piece model: {exc}", stacklevel=1)
            _model = None
    return _model


def reload_model() -> None:
    """
    Invalidate the in-memory model cache so the next inference call reloads
    ``piece_model.joblib`` from disk.  Call this after training a new model.
    """
    global _model
    _model = None
    logger.info("Piece classifier model cache cleared; will reload from disk on next call.")


# ---------------------------------------------------------------------------
# HOG feature extraction
# ---------------------------------------------------------------------------

def _hog_features(gray: np.ndarray) -> np.ndarray:
    """Extract HOG feature vector from a grayscale 100×100 square image."""
    resized = cv2.resize(gray, (100, 100))
    features = sk_hog(
        resized,
        orientations=9,
        pixels_per_cell=(10, 10),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_SIZE = 800        # Expected side length of the warped board image (px)
SQUARE_SIZE = 100       # Each of the 64 squares is 100×100 px
GRID_CELLS = 8

# Thresholds (tunable)
EMPTY_VARIANCE_THRESHOLD = 300    # Below this variance → likely empty square
EMPTY_EDGE_THRESHOLD = 0.03       # Below this edge-pixel ratio → likely empty
DARK_PIECE_THRESHOLD = 110        # Centre brightness below this → black piece
TALL_ASPECT_THRESHOLD = 0.55      # Blob height/width ratio above this → tall piece
HIGH_EDGE_THRESHOLD = 0.12        # Edge-pixel ratio above this → complex shape
CIRCULAR_THRESHOLD = 0.65         # Circularity score above this → rounded top
DIAGONAL_THRESHOLD = 0.45         # Diagonal edge ratio above this → bishop-like


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _extract_square(board_img: np.ndarray, row: int, col: int) -> np.ndarray:
    """Return the 100×100 BGR patch for grid position (row, col)."""
    y = row * SQUARE_SIZE
    x = col * SQUARE_SIZE
    return board_img[y : y + SQUARE_SIZE, x : x + SQUARE_SIZE]


def _is_empty(square_bgr: np.ndarray) -> bool:
    """
    Determine whether a square is empty based on pixel variance and edge
    density.  Empty squares are relatively uniform in colour/texture.
    """
    gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)

    # Check variance — an empty square has low variance
    variance = float(np.var(gray))
    if variance < EMPTY_VARIANCE_THRESHOLD:
        return True

    # Check edge density — an empty square has few strong edges
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / edges.size
    return edge_ratio < EMPTY_EDGE_THRESHOLD


def _piece_color(square_bgr: np.ndarray) -> str:
    """
    Determine piece colour ('white' or 'black') by examining the average
    brightness of the centre 40×40 region of the square.
    """
    cy, cx = SQUARE_SIZE // 2, SQUARE_SIZE // 2
    margin = 20
    centre = square_bgr[cy - margin : cy + margin, cx - margin : cx + margin]
    gray_centre = cv2.cvtColor(centre, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray_centre))
    return "white" if brightness >= DARK_PIECE_THRESHOLD else "black"


def _edge_density(gray: np.ndarray) -> float:
    """Return the ratio of edge pixels to total pixels."""
    edges = cv2.Canny(gray, 50, 150)
    return np.count_nonzero(edges) / edges.size


def _blob_aspect_ratio(gray: np.ndarray) -> float:
    """
    Find the largest blob in the square and return its height/width ratio.
    Returns 1.0 if no blob is found.
    """
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    largest = max(contours, key=cv2.contourArea)
    _, _, w, h = cv2.boundingRect(largest)
    if w == 0:
        return 1.0
    return h / w


def _circularity_top(gray: np.ndarray) -> float:
    """
    Measure circularity of the largest contour in the *top half* of the square.
    circularity = 4π·area / perimeter²   (1.0 = perfect circle)
    Returns 0.0 if no suitable contour is found.
    """
    top_half = gray[: SQUARE_SIZE // 2, :]
    _, thresh = cv2.threshold(top_half, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    peri = cv2.arcLength(largest, True)
    if peri == 0:
        return 0.0
    return (4 * np.pi * area) / (peri ** 2)


def _diagonal_edge_ratio(gray: np.ndarray) -> float:
    """
    Estimate how 'diagonal' the edges are by comparing Sobel responses in
    the diagonal (45°/135°) direction vs the total gradient magnitude.
    A bishop tends to have strong diagonal edges from its tall tapered shape.
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    diag1 = np.abs(sobelx + sobely)   # 45° direction
    diag2 = np.abs(sobelx - sobely)   # 135° direction
    total = np.abs(sobelx) + np.abs(sobely) + 1e-6
    diag_response = (diag1 + diag2) / (2 * total)
    return float(np.mean(diag_response))


# ---------------------------------------------------------------------------
# Piece-type heuristic
# ---------------------------------------------------------------------------

def _classify_piece_type(square_bgr: np.ndarray) -> str:
    """
    Classify the type of piece in the square using edge/blob/shape heuristics.

    Returns one of: 'K', 'Q', 'R', 'B', 'N', 'P'  (type only, not colour).

    # TODO: Replace with ML model (e.g. YOLOv8 via ultralytics) for higher
    #        accuracy.  The current approach is an intentional heuristic MVP.
    """
    gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)

    edge_den = _edge_density(gray)
    aspect = _blob_aspect_ratio(gray)
    circularity = _circularity_top(gray)
    diag_ratio = _diagonal_edge_ratio(gray)

    # --- Pawn: small, compact, low edge density, roughly square aspect ---
    if edge_den < 0.08 and aspect < 1.3:
        return "P"

    # --- Knight: medium edge density, irregular blob (low circularity, non-tall) ---
    if edge_den < HIGH_EDGE_THRESHOLD and circularity < CIRCULAR_THRESHOLD and aspect < TALL_ASPECT_THRESHOLD:
        return "N"

    # --- Bishop: tall-ish with strong diagonal edges ---
    if diag_ratio > DIAGONAL_THRESHOLD and aspect >= TALL_ASPECT_THRESHOLD:
        return "B"

    # --- King / Queen: rounded top (high circularity) and high edge density ---
    if circularity >= CIRCULAR_THRESHOLD and edge_den >= HIGH_EDGE_THRESHOLD:
        # Queen tends to be slightly wider/more complex than King
        if edge_den > 0.18:
            return "Q"
        return "K"

    # --- Rook: tall, high edge density, non-circular (crenellated top) ---
    if aspect >= TALL_ASPECT_THRESHOLD and edge_den >= HIGH_EDGE_THRESHOLD:
        return "R"

    # --- Fallback ---
    return "P"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_pieces(board_img: np.ndarray) -> list[list[Optional[str]]]:
    """
    Classify all pieces on an 800×800 warped board image.

    Parameters
    ----------
    board_img : np.ndarray
        BGR image of shape (800, 800, 3) produced by board_detector.detect_board.

    Returns
    -------
    list[list[str | None]]
        8×8 grid (row 0 = rank 8, col 0 = file a).
        Each cell is a piece code ('K','Q','R','B','N','P' for white;
        lowercase for black) or None for an empty square.
    """
    if board_img.shape[:2] != (BOARD_SIZE, BOARD_SIZE):
        board_img = cv2.resize(board_img, (BOARD_SIZE, BOARD_SIZE))

    model = _load_model()
    if model is not None:
        logger.debug("classify_pieces: using trained SVM model for inference.")
    else:
        logger.debug("classify_pieces: no model available, falling back to heuristic classifier.")

    piece_map: list[list[Optional[str]]] = []

    for row in range(GRID_CELLS):
        rank_row: list[Optional[str]] = []
        for col in range(GRID_CELLS):
            square = _extract_square(board_img, row, col)

            if _is_empty(square):
                rank_row.append(None)
            else:
                if model is not None:
                    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
                    feat = _hog_features(gray).reshape(1, -1)
                    safe = model.predict(feat)[0]
                    rank_row.append(_fen_from_safe_label(safe))
                else:
                    # Heuristic fallback
                    color = _piece_color(square)
                    piece_type = _classify_piece_type(square)
                    code = piece_type if color == "white" else piece_type.lower()
                    rank_row.append(code)

        piece_map.append(rank_row)

    return piece_map
