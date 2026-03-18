"""
collect_training_data.py
------------------------
Helper module for extracting labeled square images from a warped board image.

This module is intentionally standalone — it does NOT import from
piece_classifier.py to avoid circular imports.
"""

import os
import uuid

import cv2
import numpy as np

# Square size must match the board detector output (800x800 → 8×8 × 100×100)
_SQUARE_SIZE = 100
_GRID_CELLS = 8
_BOARD_SIZE = _SQUARE_SIZE * _GRID_CELLS  # 800


def _parse_fen_to_grid(fen: str) -> list[list[str]]:
    """
    Parse the piece-placement field of a FEN string into an 8×8 grid of labels.

    Row 0 → rank 8 (back rank for black), row 7 → rank 1 (back rank for white).
    Uppercase letters = white pieces (K, Q, R, B, N, P).
    Lowercase letters = black pieces (k, q, r, b, n, p).
    "empty" = empty square.
    """
    placement = fen.strip().split()[0]
    grid: list[list[str]] = []
    for rank_str in placement.split("/"):
        row: list[str] = []
        for ch in rank_str:
            if ch.isdigit():
                row.extend(["empty"] * int(ch))
            else:
                row.append(ch)
        if len(row) != _GRID_CELLS:
            raise ValueError(
                f"FEN rank '{rank_str}' parsed to {len(row)} cells; expected 8."
            )
        grid.append(row)
    if len(grid) != _GRID_CELLS:
        raise ValueError(
            f"FEN has {len(grid)} ranks; expected 8."
        )
    return grid


def collect_squares(
    board_img: np.ndarray,
    fen: str,
    output_dir: str = "training_data",
) -> int:
    """
    Given an 800×800 warped board image and a FEN string, extract all 64
    squares and save them as labeled PNG images.

    Parameters
    ----------
    board_img : np.ndarray
        BGR image of shape (800, 800, 3).
    fen : str
        FEN string for the current board position.  Only the piece-placement
        field (first token) is used.
    output_dir : str
        Directory under which labeled sub-folders are created.  Defaults to
        ``"training_data"`` (relative to the current working directory).

    Returns
    -------
    int
        Number of square images saved (up to 64).
    """
    # Resize defensively in case the board is not exactly 800×800
    if board_img.shape[:2] != (_BOARD_SIZE, _BOARD_SIZE):
        board_img = cv2.resize(board_img, (_BOARD_SIZE, _BOARD_SIZE))

    grid = _parse_fen_to_grid(fen)
    saved = 0

    for row in range(_GRID_CELLS):
        for col in range(_GRID_CELLS):
            label = grid[row][col]  # e.g. "P", "p", "empty"

            # Extract the 100×100 patch
            y = row * _SQUARE_SIZE
            x = col * _SQUARE_SIZE
            patch = board_img[y : y + _SQUARE_SIZE, x : x + _SQUARE_SIZE]

            # Ensure patch is exactly 100×100 (handles edge rounding)
            patch = cv2.resize(patch, (_SQUARE_SIZE, _SQUARE_SIZE))

            # Create label directory
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            # Save with a unique filename
            filename = os.path.join(label_dir, f"{uuid.uuid4().hex}.png")
            cv2.imwrite(filename, patch)
            saved += 1

    return saved
