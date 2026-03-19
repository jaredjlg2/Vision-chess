# ♟ Vision Chess

> Upload a photo of a physical chess board and instantly get a **fully playable digital position**.

---

## Screenshot

![Vision Chess UI placeholder](https://via.placeholder.com/800x450?text=Vision+Chess+Screenshot)

---

## How It Works

```
[User uploads photo]
        │
        ▼
[FastAPI backend receives image]
        │
        ▼
[board_detector.py]
  • Convert to grayscale
  • Canny edge detection
  • Find largest quadrilateral contour (the board)
  • Perspective-warp to 800×800 top-down view
        │
        ▼
[piece_classifier.py]
  • Divide into 8×8 grid (64 × 100×100 squares)
  • Per square: detect empty/occupied via variance & edge density
  • Colour detection (brightness heuristic)
  • Piece-type detection (edge density, blob shape, circularity)
        │
        ▼
[fen_builder.py]
  • Assemble FEN position string from the 8×8 piece map
        │
        ▼
[Frontend — chessboard.js + chess.js]
  • Render interactive board from FEN
  • Legal-move-only drag-and-drop
  • Checkmate / stalemate detection
```

---

## Prerequisites

- **Docker & Docker Compose** (recommended), **or**
- **Python 3.11+** for manual backend setup

---

## Quick Start with Docker

```bash
git clone https://github.com/jaredjlg2/Vision-chess.git
cd Vision-chess
docker-compose up --build
```

Then open **http://localhost** in your browser.

---

## Manual Start (without Docker)

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at **http://localhost:8000**.

### Frontend

1. Open `frontend/index.html` directly in your browser **or** serve it with any static file server.
2. The `API_URL` constant at the top of `frontend/app.js` defaults to `''` (relative URL, works via nginx).  
   For direct browser access change it to `http://localhost:8000`.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check — returns `{"status":"ok"}` |
| `POST` | `/api/detect` | Upload an image; returns `{"success":true,"fen":"...","message":"..."}` |
| `POST` | `/api/collect-training-data` | Save labeled square images from a board photo + FEN |
| `POST` | `/api/train` | Train the HOG+SVM classifier from collected data |

The `/api/detect` and `/api/collect-training-data` endpoints accept `multipart/form-data` with a field named `file`.

---

## Training the Piece Classifier

### 1. Collect Training Data

The `auto_collect.py` script renders chess-board images from FEN strings and
saves labeled square images into `backend/training_data/`.

```bash
cd backend

# Process all FENs in the default data file
python auto_collect.py

# Process the first 100 FENs
python auto_collect.py -n 100

# Process FENs 50-149
python auto_collect.py -n 100 -s 50

# Generate 3 different board styles per FEN (more variety)
python auto_collect.py --styles-per-fen 3 -n 50

# Force a specific piece style
python auto_collect.py --piece-style unicode
```

You can also collect data from a real board photo via the API:

```bash
curl -X POST http://localhost:8000/api/collect-training-data \
  -F "file=@/path/to/board_photo.jpg" \
  -F "fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

### 2. Train the Classifier

Once training data has been collected, run:

```bash
cd backend
python train_classifier.py
```

This trains a HOG + SVM model and saves it as `backend/piece_model.joblib`.
The backend will automatically use the new model on the next inference call
(no restart required).

**Prerequisites:**

```bash
pip install scikit-learn scikit-image joblib
```

---

## Limitations & Future Work

| Area | Current State | Planned Improvement |
|------|---------------|---------------------|
| Piece classification | Heuristic (edge density, blob shape) | Replace with YOLOv8 / fine-tuned CNN |
| Board detection | Contour-based (needs good lighting) | Deep-learning corner detector |
| Piece colour | Brightness threshold | Segment piece from board background |
| Turn detection | Always set to White to move | Infer from uploaded position context |
| En-passant / castling | Defaults to full rights | Detect from board history |

> The `# TODO: Replace with ML model for higher accuracy` comments in
> `piece_classifier.py` mark the exact integration points.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Install backend dev dependencies: `pip install -r backend/requirements.txt`
3. Run the backend locally: `uvicorn main:app --reload` from `backend/`
4. Open a pull request describing your changes.

All contributions welcome — especially improvements to the piece-classification accuracy!