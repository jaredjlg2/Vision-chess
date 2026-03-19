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
    python auto_collect.py --piece-style unicode    # force a specific style
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
from piece_sets import (  # noqa: E402
    PieceStyle,
    UnicodePieceStyle,
    discover_piece_styles,
    get_random_style,
)

# ---------------------------------------------------------------------------
# Board rendering constants
# ---------------------------------------------------------------------------
BOARD_PX = 800
SQUARE_PX = BOARD_PX // 8  # 100

# Unicode chess symbols (kept for reference; rendering is now in piece_sets.py)
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
    piece_style: Optional["PieceStyle"] = None,
    scale_jitter: float = 0.0,
    pos_jitter: int = 0,
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
        Pre-loaded ``ImageFont`` to use for piece glyphs when the Unicode style
        is active.  Loaded on-demand if *None*.
    piece_style:
        A :class:`~piece_sets.PieceStyle` instance that handles piece drawing.
        If *None*, the Unicode style is used (original behavior).
    scale_jitter:
        Maximum fractional scale variation applied per board (e.g. 0.15 means
        the piece size will be randomly multiplied by a factor in [0.85, 1.15]).
        ``0.0`` disables jitter.
    pos_jitter:
        Maximum pixel offset (±) applied to each piece's position independently
        to simulate imperfect centering.  ``0`` disables positional jitter.

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

    # Resolve piece style: fall back to UnicodePieceStyle if not provided
    if piece_style is None:
        if font is None:
            font = _find_unicode_font(size=80)
        piece_style = UnicodePieceStyle(font=font)

    # Per-board scale factor
    if scale_jitter > 0:
        scale_factor = random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
    else:
        scale_factor = 1.0

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

            # -- Draw piece --------------------------------------------------
            square = chess.square(chess_file, chess_rank)
            piece = board.piece_at(square)
            if piece is None:
                continue

            # Per-piece position jitter
            offset_x = random.randint(-pos_jitter, pos_jitter) if pos_jitter > 0 else 0
            offset_y = random.randint(-pos_jitter, pos_jitter) if pos_jitter > 0 else 0

            piece_style.draw_piece(
                img, draw, piece, x0, y0, SQUARE_PX,
                scale_factor=scale_factor,
                offset_x=offset_x,
                offset_y=offset_y,
            )

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
    parser.add_argument(
        "--piece-style",
        default=None,
        metavar="NAME",
        help=(
            "Force a specific piece style by name (e.g. 'cburnett', 'unicode'). "
            "Default is random selection from all discovered styles."
        ),
    )
    parser.add_argument(
        "--scale-jitter",
        type=float,
        default=0.15,
        metavar="F",
        help=(
            "Maximum fractional scale jitter applied per board (e.g. 0.15 → "
            "piece size varies ±15%%). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--pos-jitter",
        type=int,
        default=3,
        metavar="PX",
        help=(
            "Maximum pixel position offset (±) applied per piece to simulate "
            "imperfect centering. Set to 0 to disable."
        ),
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

    # -- Discover piece styles -----------------------------------------------
    font = _find_unicode_font(size=80)
    if font is None:
        print(
            "Warning: no TrueType font found; Unicode pieces will be rendered as "
            "single letters. Install DejaVu or Noto fonts for better output."
        )

    all_styles = discover_piece_styles(font=font)
    style_names = [s.name for s in all_styles]
    print(f"Available piece styles: {', '.join(style_names)}")

    # Resolve forced style if --piece-style was given
    forced_style = None
    if args.piece_style is not None:
        matches = [s for s in all_styles if s.name == args.piece_style]
        if not matches:
            print(
                f"Error: piece style '{args.piece_style}' not found. "
                f"Available styles: {', '.join(style_names)}"
            )
            sys.exit(1)
        forced_style = matches[0]
        print(f"Forcing piece style: {forced_style.name}")

    # -- Main loop -----------------------------------------------------------
    t0 = time.time()
    total_squares = 0
    img_counter = start * styles + 1  # global image counter for filenames

    for fen_idx, fen in enumerate(fens, start=1):
        for style_idx in range(styles):
            theme = random.choice(COLOR_THEMES)
            flip = random.random() < 0.5
            noise = random.uniform(3, 12)

            # Select piece style for this board
            piece_style = forced_style if forced_style is not None else get_random_style(all_styles)

            # Render board
            bgr = render_board(
                fen,
                theme=theme,
                flip=flip,
                noise_level=noise,
                piece_style=piece_style,
                scale_jitter=args.scale_jitter,
                pos_jitter=args.pos_jitter,
            )

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
                f" → {img_name}{flip_str}"
                f" [style={piece_style.name}]"
                f" → {squares} squares saved"
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
