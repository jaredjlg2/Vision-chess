"""
board_detector.py
-----------------
Uses OpenCV to detect a chess board in an image, find the largest quadrilateral
contour (the board), and apply a perspective transform to produce a clean
top-down 800x800 view of the board.
"""

import cv2
import numpy as np
from PIL import Image


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four corner points as: top-left, top-right, bottom-right, bottom-left.
    This is required by cv2.getPerspectiveTransform.
    """
    rect = np.zeros((4, 2), dtype="float32")
    # Top-left has the smallest sum; bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # Top-right has the smallest difference; bottom-left has the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _find_board_contour(gray: np.ndarray) -> np.ndarray | None:
    """
    Attempt to find the quadrilateral contour that represents the chess board.
    Returns the four corner points (as float32 array) or None if not found.
    """
    # Blur + Canny to find strong edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges so nearby lines connect
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = gray.shape[0] * gray.shape[1]

    for contour in contours[:10]:  # Only check the 10 largest contours
        area = cv2.contourArea(contour)
        # Skip if the contour is less than 5% of the image area (too small)
        if area < 0.05 * image_area:
            break

        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

    return None


def detect_board(image: Image.Image | np.ndarray, output_size: int = 800) -> np.ndarray:
    """
    Detect the chess board in the given image and return a top-down
    perspective-corrected view as a numpy array (BGR, uint8).

    Parameters
    ----------
    image : PIL.Image.Image or np.ndarray
        The input image containing a chess board.
    output_size : int
        The side length (in pixels) of the square output image (default 800).

    Returns
    -------
    np.ndarray
        Warped board image of shape (output_size, output_size, 3).

    Raises
    ------
    ValueError
        If no chess board quadrilateral can be found in the image.
    """
    # Convert PIL Image → numpy BGR array if necessary
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image.copy()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    corners = _find_board_contour(gray)

    if corners is None:
        raise ValueError(
            "Could not detect a chess board in the image. "
            "Please ensure the board is clearly visible and well-lit."
        )

    # Order corners: top-left, top-right, bottom-right, bottom-left
    ordered = _order_points(corners)

    # Destination corners for the output square
    dst = np.array(
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ],
        dtype="float32",
    )

    # Compute and apply the perspective transform
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img_bgr, M, (output_size, output_size))

    return warped
