"""
main.py
-------
FastAPI entry-point for Vision Chess.

Endpoints:
  GET  /health                       → health check
  POST /api/detect                   → accepts a chess-board photo; returns a FEN string
  POST /api/collect-training-data    → save labeled square images for classifier training
  POST /api/train                    → train the HOG+SVM classifier from collected data
"""

import io
import os
import traceback

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from board_detector import detect_board
from collect_training_data import collect_squares
from piece_classifier import classify_pieces, reload_model
from fen_builder import build_fen
from train_classifier import train_model

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


@app.post("/api/collect-training-data")
async def collect_training_data(
    file: UploadFile = File(...),
    fen: str = Form(
        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ),
) -> JSONResponse:
    """
    Accept a board photo and an optional FEN string, run board detection, and
    save all 64 square images as labeled PNG files under ``training_data/``.

    Response (success):
        { "success": true, "squares_saved": <n>,
          "message": "Saved <n> labeled squares to training_data/." }

    Response (error):
        { "success": false, "squares_saved": 0, "message": "<human-readable error>" }
    """
    if file.content_type and not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "squares_saved": 0,
                "message": (
                    f"Unsupported file type '{file.content_type}'. "
                    "Please upload an image."
                ),
            },
        )

    try:
        raw_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        warped = detect_board(pil_image)

        training_dir = os.path.join(os.path.dirname(__file__), "training_data")
        count = collect_squares(warped, fen, output_dir=training_dir)

        return JSONResponse(
            content={
                "success": True,
                "squares_saved": count,
                "message": f"Saved {count} labeled squares to training_data/.",
            }
        )

    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "squares_saved": 0,
                "message": str(exc),
            },
        )

    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "squares_saved": 0,
                "message": "An internal error occurred while processing the image.",
            },
        )


@app.post("/api/train")
async def train() -> JSONResponse:
    """
    Trigger training of the HOG+SVM piece classifier from the labeled images
    that have been collected via ``/api/collect-training-data``.

    After successful training the in-memory model cache is invalidated so the
    new model is used immediately — no restart required.

    Response (success):
        { "success": true, "samples": <n>, "accuracy": <float>,
          "report": "<classification report>", "message": "..." }

    Response (error):
        { "success": false, "message": "<human-readable error>" }
    """
    try:
        result = train_model()

        # Invalidate the cached model so the next inference call loads the
        # freshly trained file from disk.
        reload_model()

        return JSONResponse(
            content={
                "success": True,
                "samples": result["samples"],
                "accuracy": result["accuracy"],
                "report": result["report"],
                "message": (
                    f"Training complete. "
                    f"{result['samples']} samples, "
                    f"accuracy {result['accuracy']:.2%}. "
                    "Model is now active."
                ),
            }
        )

    except ImportError as exc:
        return JSONResponse(
            status_code=501,
            content={
                "success": False,
                "message": str(exc),
            },
        )

    except (FileNotFoundError, ValueError) as exc:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": str(exc),
            },
        )

    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "An internal error occurred during training.",
            },
        )
