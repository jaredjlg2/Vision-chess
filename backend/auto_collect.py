"""
auto_collect.py
---------------
Automated training-data collection pipeline.

Reads FEN strings from an Excel or CSV file, renders an 800×800 chess-board
image for each FEN using a pure-Pillow renderer (no cairosvg required), and
feeds the rendered image directly into ``collect_squares()`` from
``collect_training_data.py`` to produce labeled square images.

Usage
-----
    python auto_collect.py                          # process all FENs
    python auto_collect.py -n 100                   # first 100 FENs
    python auto_collect.py -n 100 -s 50             # FENs 50-149
    python auto_collect.py --styles-per-fen 3 -n 50 # 3 styles × 50 FENs
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import chess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Import existing helper (must run from the backend/ directory, or backend/
# must be on sys.path).
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).parent.resolve()
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from collect_training_data import collect_squares  # noqa: E402

# ---------------------------------------------------------------------------
# Board rendering constants
# ---------------------------------------------------------------------------
BOARD_PX = 800
SQUARE_PX = BOARD_PX // 8  # 100

# Unicode chess symbols indexed by python-chess piece symbol
_PIECE_UNICODE: dict[str, str] = {
    "K": "♔",
    "Q": "♕",
    "R": "♖",
    "B": "♗",
    "N": "♘",
    "P": "♙",
    "k": "♚",
    "q": "♛",
    "r": "♜",
    "b": "♝",
    "n": "♞",
    "p": "♟",
}

# ---------------------------------------------------------------------------
# Color themes  (light_square_RGB, dark_square_RGB)
# ---------------------------------------------------------------------------
COLOR_THEMES: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = [
    # Classic brown / cream (chess.com-ish)
    ((240, 217, 181), (181, 136, 99)),
    # Green / white (chess.com green)
    ((238, 238, 210), (118, 150, 86)),
    # Blue / white (Lichess blue)
    ((240, 240, 240), (93, 138, 168)),
    # Gray tones
    ((210, 210, 210), (130, 130, 130)),
    # Coral / ivory
    ((255, 245, 220), (200, 105, 85)),
    # Wood tones – warm mahogany
    ((234, 200, 155), (139, 90, 43)),
    # Purple / cream
    ((245, 240, 255), (130, 90, 160)),
    # Teal / white
    ((230, 255, 250), (60, 160, 140)),
    # Navy / gold
    ((255, 240, 190), (35, 60, 110)),
    # Slate blue / light lavender
    ((220, 220, 245), (85, 85, 150)),
]


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

def _find_unicode_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    """Try to load a TrueType font that supports chess Unicode symbols."""
    candidates = [
        # Linux / CI
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        # Windows
        r"C:\Windows\Fonts\seguisym.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Last resort: let Pillow search its default paths
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Board renderer
# ---------------------------------------------------------------------------

def render_board(
    fen: str,
    *,
    theme: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
    flip: bool = False,
    noise_level: float = 0.0,
    font: Optional[ImageFont.FreeTypeFont] = None,
) -> np.ndarray:
    """Render an 800×800 chess board from a FEN string using Pillow.

    Parameters
    ----------
    fen:
        FEN string describing the board position.
    theme:
        ``(light_rgb, dark_rgb)`` color pair.  A random theme is chosen if
        *None*.
    flip:
        When *True* the board is rendered from Black's perspective (rank 1 at
        the top).
    noise_level:
        Standard deviation (0–255) of Gaussian noise added to the board.  Use
        a small value like 5–15 for subtle variation.
    font:
        Pre-loaded ``ImageFont`` to use for piece glyphs.  Loaded on-demand if
        *None*.

    Returns
    -------
    np.ndarray
        BGR image of shape ``(800, 800, 3)`` ready for ``collect_squares()``.
    """
    if theme is None:
        theme = random.choice(COLOR_THEMES)
    light_color, dark_color = theme

    board = chess.Board(fen)
    img = Image.new("RGB", (BOARD_PX, BOARD_PX))
    draw = ImageDraw.Draw(img)

    if font is None:
        font = _find_unicode_font(size=80)

    # -- Draw squares --------------------------------------------------------
    for rank_idx in range(8):  # 0 = top row of the image
        for file_idx in range(8):  # 0 = left column
            # Map image row/col to chess rank/file
            if flip:
                chess_rank = rank_idx       # rank 0 (rank 1) at top
                chess_file = 7 - file_idx  # file h at left
            else:
                chess_rank = 7 - rank_idx  # rank 7 (rank 8) at top
                chess_file = file_idx       # file a at left

            # Square colour: light when (rank + file) is even
            is_light = (chess_rank + chess_file) % 2 == 0
            sq_color = light_color if is_light else dark_color

            x0 = file_idx * SQUARE_PX
            y0 = rank_idx * SQUARE_PX
            x1 = x0 + SQUARE_PX - 1
            y1 = y0 + SQUARE_PX - 1
            draw.rectangle([x0, y0, x1, y1], fill=sq_color)

            # -- Draw piece glyph -------------------------------------------
            square = chess.square(chess_file, chess_rank)
            piece = board.piece_at(square)
            if piece is None:
                continue

            glyph = _PIECE_UNICODE.get(piece.symbol())
            if glyph is None:
                continue

            if font is not None:
                # Determine text colour: white pieces → near-white with dark
                # shadow; black pieces → near-black with light shadow.
                if piece.color == chess.WHITE:
                    text_color = (255, 255, 255)
                    shadow_color = (40, 40, 40)
                else:
                    text_color = (20, 20, 20)
                    shadow_color = (220, 220, 220)

                # Centre the glyph on the square using textbbox
                bbox_offset_x = 0
                bbox_offset_y = 0
                try:
                    bbox = draw.textbbox((0, 0), glyph, font=font)
                    glyph_w = bbox[2] - bbox[0]
                    glyph_h = bbox[3] - bbox[1]
                    bbox_offset_x = bbox[0]
                    bbox_offset_y = bbox[1]
                except AttributeError:
                    # Pillow <9.2 fallback: getbbox returns (left,top,right,bottom)
                    fb = font.getbbox(glyph)  # type: ignore[attr-defined]
                    glyph_w = fb[2] - fb[0]
                    glyph_h = fb[3] - fb[1]
                    bbox_offset_x = fb[0]
                    bbox_offset_y = fb[1]

                cx = x0 + (SQUARE_PX - glyph_w) // 2 - bbox_offset_x
                cy = y0 + (SQUARE_PX - glyph_h) // 2 - bbox_offset_y

                # Shadow pass (offset by 2px)
                draw.text((cx + 2, cy + 2), glyph, font=font, fill=shadow_color)
                # Main glyph
                draw.text((cx, cy), glyph, font=font, fill=text_color)
            else:
                # Fallback: draw FEN letter when no font is available
                fallback_font = ImageFont.load_default()
                label = piece.symbol().upper() if piece.color == chess.WHITE else piece.symbol().lower()
                fg = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                draw.text((x0 + 5, y0 + 5), label, font=fallback_font, fill=fg)

    # -- Optional noise ------------------------------------------------------
    if noise_level > 0:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, noise_level, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # -- Convert to BGR numpy array ------------------------------------------
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr


# ---------------------------------------------------------------------------
# FEN file reader
# ---------------------------------------------------------------------------

def load_fens(path: str) -> List[str]:
    """Load FEN strings from an Excel (.xlsx) or CSV file.

    The FEN column is identified by looking for a header named ``fen`` or
    ``FEN`` (case-insensitive).  If no such header exists the first column is
    used.  Empty rows are skipped.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"FEN file not found: {p}")

    suffix = p.suffix.lower()
    rows: List[List[str]]

    if suffix in (".xlsx", ".xls"):
        try:
            import openpyxl
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required to read .xlsx files. "
                "Install it with:  pip install openpyxl"
            ) from exc

        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
        ws = wb.active
        rows = [[str(cell.value) if cell.value is not None else "" for cell in row] for row in ws.iter_rows()]
        wb.close()

    elif suffix == ".csv":
        import csv

        with open(p, newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            rows = [row for row in reader]
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}.  Use .xlsx or .csv.")

    if not rows:
        return []

    # Detect FEN column index
    fen_col = 0
    header_row = rows[0]
    for idx, cell in enumerate(header_row):
        if cell.strip().lower() == "fen":
            fen_col = idx
            break

    # Determine if first row is a header (contains the word "fen" or is
    # non-FEN text)
    first_is_header = any(cell.strip().lower() == "fen" for cell in header_row)

    start_row = 1 if first_is_header else 0
    fens: List[str] = []
    for row in rows[start_row:]:
        if fen_col >= len(row):
            continue
        val = row[fen_col].strip()
        if val and val.lower() != "none":
            fens.append(val)

    return fens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated chess training-data collection from a FEN list.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fen-file", "-f",
        default=str(Path("~/Documents/fens.xlsx").expanduser()),
        metavar="PATH",
        help="Path to the Excel (.xlsx) or CSV file containing FEN strings.",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=None,
        metavar="N",
        help="Number of FENs to process (default: all).",
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        metavar="IDX",
        help="0-based index of the first FEN to process.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=str(_BACKEND_DIR / "board_images"),
        metavar="DIR",
        help="Directory where rendered board PNGs are saved.",
    )
    parser.add_argument(
        "--training-dir", "-t",
        default=str(_BACKEND_DIR / "training_data"),
        metavar="DIR",
        help="Directory for labeled training-square sub-folders.",
    )
    parser.add_argument(
        "--styles-per-fen",
        type=int,
        default=1,
        metavar="N",
        help="Number of different board-style renderings per FEN.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # -- Load FENs -----------------------------------------------------------
    print(f"Reading FENs from: {args.fen_file}")
    all_fens = load_fens(args.fen_file)
    if not all_fens:
        print("No FENs found in the file. Exiting.")
        sys.exit(1)

    start = args.start
    end = start + args.count if args.count is not None else len(all_fens)
    fens = all_fens[start:end]

    if not fens:
        print(f"No FENs in the selected range [{start}:{end}]. Exiting.")
        sys.exit(1)

    total_fens = len(fens)
    styles = max(1, args.styles_per_fen)

    print(
        f"Processing {total_fens} FEN(s) × {styles} style(s) = "
        f"{total_fens * styles} board image(s)"
    )

    # -- Prepare output directories -----------------------------------------
    output_dir = Path(args.output_dir)
    training_dir = Path(args.training_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    # -- Pre-load font once --------------------------------------------------
    font = _find_unicode_font(size=80)
    if font is None:
        print(
            "Warning: no TrueType font found; pieces will be rendered as "
            "single letters. Install DejaVu or Noto fonts for better output."
        )

    # -- Main loop -----------------------------------------------------------
    t0 = time.time()
    total_squares = 0
    img_counter = start * styles + 1  # global image counter for filenames

    for fen_idx, fen in enumerate(fens, start=1):
        for style_idx in range(styles):
            theme = random.choice(COLOR_THEMES)
            flip = random.random() < 0.5
            noise = random.uniform(3, 12)

            # Render board
            bgr = render_board(fen, theme=theme, flip=flip, noise_level=noise, font=font)

            # Save board image
            img_name = f"chess{img_counter:04d}.png"
            img_path = output_dir / img_name
            cv2.imwrite(str(img_path), bgr)

            # Chop into 64 labeled squares.
            # collect_squares expects standard orientation (rank 8 at top,
            # file a at left). Rotate 180° to restore that when flip=True.
            if flip:
                # Restore standard orientation for collect_squares
                collect_img = cv2.rotate(bgr, cv2.ROTATE_180)
            else:
                collect_img = bgr

            squares = collect_squares(collect_img, fen, output_dir=str(training_dir))
            total_squares += squares

            flip_str = " (flipped)" if flip else ""
            print(
                f"[{fen_idx}/{total_fens}] FEN: {fen[:50]}{'…' if len(fen) > 50 else ''}"
                f" → {img_name}{flip_str} → {squares} squares saved"
            )

            img_counter += 1

    elapsed = time.time() - t0
    print(
        f"\nDone! Processed {total_fens} FEN(s), {total_fens * styles} image(s), "
        f"{total_squares} total square images saved in {elapsed:.1f}s."
    )
    print(f"Board images:   {output_dir}")
    print(f"Training data:  {training_dir}")


if __name__ == "__main__":
    main()
