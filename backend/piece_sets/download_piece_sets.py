#!/usr/bin/env python3
"""
download_piece_sets.py
----------------------
Downloads open-license chess piece SVG sets from Lichess's GitHub repository
into the correct directory structure under ``backend/piece_sets/``.

After downloading the SVGs, optionally rasterizes them to 200×200 PNG files
using cairosvg (if installed).  The PNGs are what the renderer actually loads
at runtime; the SVGs are kept as the canonical source.

Usage
-----
    python backend/piece_sets/download_piece_sets.py
    python backend/piece_sets/download_piece_sets.py --no-rasterize
    python backend/piece_sets/download_piece_sets.py --sets cburnett merida
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece"

PIECE_SETS = ["cburnett", "merida", "pirouetti", "alpha"]

PIECE_FILES = [
    "wK", "wQ", "wR", "wB", "wN", "wP",
    "bK", "bQ", "bR", "bB", "bN", "bP",
]

# Directory that contains this script
_THIS_DIR = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_svg(set_name: str, piece: str, dest_dir: Path) -> bool:
    url = f"{BASE_URL}/{set_name}/{piece}.svg"
    dest = dest_dir / f"{piece}.svg"
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:
        print(f"  ERROR downloading {url}: {exc}", file=sys.stderr)
        return False


def rasterize_png(svg_path: Path, png_path: Path, size: int = 200) -> bool:
    try:
        import cairosvg  # type: ignore[import]
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                         output_width=size, output_height=size)
        return True
    except ImportError:
        print("  cairosvg not installed; skipping PNG rasterization.", file=sys.stderr)
        print("  Install it with:  pip install cairosvg", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"  ERROR rasterizing {svg_path}: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Lichess SVG piece sets and rasterize to PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sets", nargs="+", default=PIECE_SETS,
        choices=PIECE_SETS,
        help="Which piece sets to download.",
    )
    parser.add_argument(
        "--no-rasterize", action="store_true",
        help="Skip PNG rasterization (SVGs only).",
    )
    parser.add_argument(
        "--size", type=int, default=200,
        help="Output PNG size in pixels (square).",
    )
    args = parser.parse_args(argv)

    ok = True
    for set_name in args.sets:
        dest_dir = _THIS_DIR / set_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{set_name}]")

        for piece in PIECE_FILES:
            svg_path = dest_dir / f"{piece}.svg"
            png_path = dest_dir / f"{piece}.png"

            # Download SVG
            if svg_path.exists():
                print(f"  {piece}.svg  (already exists, skipping)")
            else:
                if download_svg(set_name, piece, dest_dir):
                    print(f"  {piece}.svg  downloaded")
                else:
                    ok = False
                    continue

            # Rasterize to PNG
            if args.no_rasterize:
                continue
            if png_path.exists():
                print(f"  {piece}.png  (already exists, skipping)")
            else:
                if rasterize_png(svg_path, png_path, size=args.size):
                    print(f"  {piece}.png  rasterized ({os.path.getsize(png_path)} bytes)")
                else:
                    ok = False

    print("\nDone." if ok else "\nCompleted with some errors.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
