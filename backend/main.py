"""
main.py
-------
FastAPI entry-point for Vision Chess.

Endpoints:
  GET  /health          → health check
  POST /api/detect      → accepts a chess-board photo; returns a FEN string
"""

import io
import traceback

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from board_detector import detect_board
from piece_classifier import classify_pieces
from fen_builder import build_fen

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Vision Chess API", version="1.0.0")

# Enable CORS so the frontend (served from any origin) can talk to this API.
# NOTE: In production, restrict allow_origins to your specific frontend domain
# instead of using the wildcard ["*"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept a multipart image upload, run the computer-vision pipeline, and
    return the detected board position as a FEN string.

    Response (success):
        { "success": true, "fen": "<fen>", "message": "Board detected successfully." }

    Response (error):
        { "success": false, "fen": null, "message": "<human-readable error>" }
    """
    # --- Validate file type ---
    if file.content_type and not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "fen": None,
                "message": f"Unsupported file type '{file.content_type}'. Please upload an image.",
            },
        )

    try:
        # --- Read uploaded bytes → PIL Image ---
        raw_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        # --- Stage 1: Detect & warp the board ---
        warped = detect_board(pil_image)

        # --- Stage 2: Classify pieces per square ---
        piece_map = classify_pieces(warped)

        # --- Stage 3: Build FEN string ---
        fen = build_fen(piece_map)

        return JSONResponse(
            content={
                "success": True,
                "fen": fen,
                "message": "Board detected successfully.",
            }
        )

    except ValueError as exc:
        # Raised by board_detector when no board is found
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "fen": None,
                "message": str(exc),
            },
        )

    except Exception:
        # Catch-all — log the traceback server-side but return a clean message
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "fen": None,
                "message": "An internal error occurred while processing the image.",
            },
        )
