"""
piece_sets.py
-------------
Piece-style discovery, loading, and rendering for auto_collect.py.

Supports two rendering strategies:

1. **SvgPieceStyle** – loads pre-rasterized PNG images (200×200) from a
   directory that follows the Lichess naming convention (wK.svg / wK.png,
   bQ.png, etc.) and composites them onto the board using Pillow.

2. **UnicodePieceStyle** – the original behavior: draws Unicode chess glyphs
   (♔♕♖…) using a system TrueType font.

Usage::

    from piece_sets import discover_piece_styles, get_random_style

    styles = discover_piece_styles()   # called once at startup
    style  = get_random_style(styles)  # one per board render
    style.draw_piece(img, draw, piece, x0, y0, square_size,
                     scale_factor=1.05, offset_x=1, offset_y=-2)
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
_PIECE_SETS_DIR = Path(__file__).parent.resolve() / "piece_sets"

# Mapping from python-chess piece symbol to Lichess filename stem
# python-chess uses uppercase letters for White, lowercase for Black.
_SYMBOL_TO_FILENAME: Dict[str, str] = {
    "K": "wK", "Q": "wQ", "R": "wR", "B": "wB", "N": "wN", "P": "wP",
    "k": "bK", "q": "bQ", "r": "bR", "b": "bB", "n": "bN", "p": "bP",
}

# All 12 expected piece filenames (stems without extension)
_EXPECTED_STEMS = list(_SYMBOL_TO_FILENAME.values())

# Unicode chess glyphs (retained for UnicodePieceStyle)
_PIECE_UNICODE: Dict[str, str] = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class PieceStyle(ABC):
    """Abstract interface for drawing a chess piece onto a PIL Image."""

    name: str

    @abstractmethod
    def draw_piece(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        piece: chess.Piece,
        x0: int,
        y0: int,
        square_size: int,
        scale_factor: float = 1.0,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        """Draw *piece* onto *img* with the square's top-left corner at (x0, y0)."""


# ---------------------------------------------------------------------------
# SVG / PNG image-based style
# ---------------------------------------------------------------------------

class SvgPieceStyle(PieceStyle):
    """Renders pieces from pre-rasterized PNG files (200×200 px each).

    The directory is expected to contain files named ``wK.png``, ``bK.png``,
    etc., matching the Lichess naming convention.  An optional SVG file
    (``wK.svg``) may exist alongside the PNG but is not used at runtime.
    """

    def __init__(self, directory: Path) -> None:
        self.name = directory.name
        self._dir = directory
        # Cache: symbol → PIL Image (RGBA, 200×200)
        self._cache: Dict[str, Image.Image] = {}

    def _load(self, symbol: str) -> Optional[Image.Image]:
        if symbol in self._cache:
            return self._cache[symbol]
        stem = _SYMBOL_TO_FILENAME.get(symbol)
        if stem is None:
            return None
        png_path = self._dir / f"{stem}.png"
        if not png_path.exists():
            return None
        try:
            img = Image.open(png_path).convert("RGBA")
            self._cache[symbol] = img
            return img
        except Exception:
            return None

    def draw_piece(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,  # noqa: ARG002  (unused but required by interface)
        piece: chess.Piece,
        x0: int,
        y0: int,
        square_size: int,
        scale_factor: float = 1.0,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        piece_img = self._load(piece.symbol())
        if piece_img is None:
            return

        # Scale piece image to (square_size × scale_factor)
        target_size = max(1, int(square_size * scale_factor))
        resized = piece_img.resize((target_size, target_size), Image.LANCZOS)

        # Centre within the square + jitter offset
        paste_x = x0 + (square_size - target_size) // 2 + offset_x
        paste_y = y0 + (square_size - target_size) // 2 + offset_y

        # Composite using alpha channel as mask
        img.paste(resized, (paste_x, paste_y), resized)


# ---------------------------------------------------------------------------
# Unicode glyph style
# ---------------------------------------------------------------------------

class UnicodePieceStyle(PieceStyle):
    """Renders pieces using Unicode chess glyphs and a TrueType font."""

    name = "unicode"

    def __init__(self, font: Optional[ImageFont.FreeTypeFont] = None) -> None:
        self._font = font  # may be None → use fallback

    def draw_piece(
        self,
        img: Image.Image,  # noqa: ARG002
        draw: ImageDraw.ImageDraw,
        piece: chess.Piece,
        x0: int,
        y0: int,
        square_size: int,
        scale_factor: float = 1.0,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        glyph = _PIECE_UNICODE.get(piece.symbol())
        if glyph is None:
            return

        if self._font is not None:
            font = self._font
            # Reload at a scaled size if scale_factor != 1.0
            if scale_factor != 1.0:
                base_size = getattr(font, "size", 80)
                scaled_size = max(8, int(base_size * scale_factor))
                try:
                    font = ImageFont.truetype(font.path, scaled_size)
                except Exception:
                    pass  # Keep original font if resizing fails

            if piece.color == chess.WHITE:
                text_color: Tuple[int, int, int] = (255, 255, 255)
                shadow_color: Tuple[int, int, int] = (40, 40, 40)
            else:
                text_color = (20, 20, 20)
                shadow_color = (220, 220, 220)

            bbox_offset_x = 0
            bbox_offset_y = 0
            try:
                bbox = draw.textbbox((0, 0), glyph, font=font)
                glyph_w = bbox[2] - bbox[0]
                glyph_h = bbox[3] - bbox[1]
                bbox_offset_x = bbox[0]
                bbox_offset_y = bbox[1]
            except AttributeError:
                fb = font.getbbox(glyph)  # type: ignore[attr-defined]
                glyph_w = fb[2] - fb[0]
                glyph_h = fb[3] - fb[1]
                bbox_offset_x = fb[0]
                bbox_offset_y = fb[1]

            cx = x0 + (square_size - glyph_w) // 2 - bbox_offset_x + offset_x
            cy = y0 + (square_size - glyph_h) // 2 - bbox_offset_y + offset_y

            draw.text((cx + 2, cy + 2), glyph, font=font, fill=shadow_color)
            draw.text((cx, cy), glyph, font=font, fill=text_color)
        else:
            fallback_font = ImageFont.load_default()
            label = piece.symbol().upper() if piece.color == chess.WHITE else piece.symbol().lower()
            fg = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
            draw.text((x0 + 5 + offset_x, y0 + 5 + offset_y), label, font=fallback_font, fill=fg)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _is_valid_piece_set_dir(directory: Path) -> bool:
    """Return True if *directory* contains all 12 expected PNG piece files."""
    for stem in _EXPECTED_STEMS:
        if not (directory / f"{stem}.png").exists():
            return False
    return True


def discover_piece_styles(
    font: Optional[ImageFont.FreeTypeFont] = None,
) -> List[PieceStyle]:
    """Discover all available piece styles.

    Scans ``backend/piece_sets/`` for subdirectories that contain the full
    complement of 12 PNG piece files (``wK.png`` … ``bP.png``).  This
    includes bundled sets (cburnett, merida, etc.) **and** any subdirectories
    inside ``piece_sets/custom/`` that the user has placed there.

    Always appends a ``UnicodePieceStyle`` as the last fallback.

    Parameters
    ----------
    font:
        Pre-loaded TrueType font for the Unicode style.  If *None* the
        Unicode style will use its own internal fallback.

    Returns
    -------
    list[PieceStyle]
        At least one style (Unicode) is always returned.
    """
    styles: List[PieceStyle] = []

    if _PIECE_SETS_DIR.is_dir():
        for entry in sorted(_PIECE_SETS_DIR.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name == "custom":
                # Recurse one level into custom/
                for sub in sorted(entry.iterdir()):
                    if sub.is_dir() and _is_valid_piece_set_dir(sub):
                        styles.append(SvgPieceStyle(sub))
            elif _is_valid_piece_set_dir(entry):
                styles.append(SvgPieceStyle(entry))

    styles.append(UnicodePieceStyle(font=font))
    return styles


def get_random_style(styles: Optional[List[PieceStyle]] = None) -> PieceStyle:
    """Return a randomly chosen :class:`PieceStyle`.

    Parameters
    ----------
    styles:
        List returned by :func:`discover_piece_styles`.  If *None*, discovers
        styles on each call (less efficient; prefer passing the pre-built list).
    """
    if styles is None:
        styles = discover_piece_styles()
    return random.choice(styles)
