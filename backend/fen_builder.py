"""
fen_builder.py
--------------
Converts an 8x8 piece map (list of lists) into a valid FEN position string,
then appends default turn/castling/en-passant fields so the result is a
complete FEN that can be loaded by chess.js / python-chess.
"""

from typing import Optional


def build_fen(piece_map: list[list[Optional[str]]]) -> str:
    """
    Build a full FEN string from an 8x8 piece map.

    Parameters
    ----------
    piece_map : list[list[str | None]]
        8 rows (rank 8 → rank 1, i.e. row 0 = rank 8) × 8 columns (file a → h).
        Each cell is either a piece code string ('K','Q','R','B','N','P' for
        white, lowercase for black) or None for an empty square.

    Returns
    -------
    str
        A complete FEN string, e.g.
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    """
    ranks = []
    for row in piece_map:
        rank_str = ""
        empty_count = 0
        for cell in row:
            if cell is None:
                empty_count += 1
            else:
                if empty_count:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += cell
        if empty_count:
            rank_str += str(empty_count)
        ranks.append(rank_str)

    position = "/".join(ranks)

    # Default: white to move, all castling rights, no en-passant, move counters
    return f"{position} w KQkq - 0 1"
