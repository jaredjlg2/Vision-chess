/**
 * app.js – Vision Chess frontend logic
 *
 * Responsibilities:
 *  - Handle file drag-and-drop and file input
 *  - Show image preview
 *  - POST image to the backend /api/detect endpoint
 *  - Parse the returned FEN and initialise chessboard.js
 *  - Enforce legal-move-only drag-and-drop via chess.js
 *  - Detect checkmate / stalemate after each move
 *  - Support "Play as White" / "Play as Black" orientation toggle
 *  - Copy FEN to clipboard
 */

/* ------------------------------------------------------------------ */
/* Configuration                                                        */
/* ------------------------------------------------------------------ */

// When served via nginx (Docker Compose) the frontend and backend share the
// same origin, so a relative URL works out of the box.
// For manual local development (opening index.html directly without nginx),
// change this to 'http://localhost:8000'.
const API_URL = '';

/* ------------------------------------------------------------------ */
/* DOM references                                                       */
/* ------------------------------------------------------------------ */

const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImg     = document.getElementById('preview-img');
const detectBtn      = document.getElementById('detect-btn');
const statusMsg      = document.getElementById('status-message');
const spinner        = document.getElementById('spinner');
const uploadCard     = document.getElementById('upload-card');
const resultCard     = document.getElementById('result-card');
const fenDisplay     = document.getElementById('fen-display');
const copyFenBtn     = document.getElementById('copy-fen-btn');
const playWhiteBtn   = document.getElementById('play-white-btn');
const playBlackBtn   = document.getElementById('play-black-btn');
const resetBtn       = document.getElementById('reset-btn');
const newUploadBtn   = document.getElementById('new-upload-btn');
const moveStatus     = document.getElementById('move-status');

/* ------------------------------------------------------------------ */
/* State                                                                */
/* ------------------------------------------------------------------ */

let selectedFile   = null;   // The File object the user selected
let board          = null;   // chessboard.js instance
let game           = null;   // chess.js instance
let detectedFen    = null;   // FEN returned by the API
let playerColor    = 'w';    // 'w' = white, 'b' = black

/* ------------------------------------------------------------------ */
/* File selection helpers                                               */
/* ------------------------------------------------------------------ */

/** Show a preview of the selected file and enable the Detect button. */
function handleFileSelected(file) {
  if (!file || !file.type.startsWith('image/')) {
    showStatus('Please select a valid image file.', 'error');
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewContainer.hidden = false;
  detectBtn.disabled = false;
  showStatus('', '');
}

/* ------------------------------------------------------------------ */
/* Drag-and-drop                                                        */
/* ------------------------------------------------------------------ */

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFileSelected(fileInput.files[0]);
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFileSelected(e.dataTransfer.files[0]);
});

/* ------------------------------------------------------------------ */
/* Status display                                                       */
/* ------------------------------------------------------------------ */

function showStatus(msg, type = 'info') {
  statusMsg.textContent = msg;
  statusMsg.className = `status-message ${type}`;
}

/* ------------------------------------------------------------------ */
/* Board detection API call                                             */
/* ------------------------------------------------------------------ */

detectBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  // Show spinner, disable button
  detectBtn.disabled = true;
  spinner.hidden = false;
  showStatus('Detecting board…', 'info');

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch(`${API_URL}/api/detect`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (data.success && data.fen) {
      detectedFen = data.fen;
      showStatus('Board detected! You can now play the position.', 'success');
      initChessBoard(detectedFen);
      resultCard.hidden = false;
      resultCard.scrollIntoView({ behavior: 'smooth' });
    } else {
      showStatus(data.message || 'Detection failed. Please try another image.', 'error');
    }
  } catch (err) {
    console.error('API error:', err);
    showStatus(
      'Could not reach the backend. Make sure it is running at ' + API_URL,
      'error'
    );
  } finally {
    spinner.hidden = true;
    detectBtn.disabled = false;
  }
});

/* ------------------------------------------------------------------ */
/* chess.js + chessboard.js initialisation                             */
/* ------------------------------------------------------------------ */

/**
 * Initialise (or re-initialise) the chess board and game logic from a FEN.
 * @param {string} fen - A valid FEN string.
 */
function initChessBoard(fen) {
  // Destroy existing board instance if present
  if (board) {
    board.destroy();
    board = null;
  }

  // Create new chess.js game from the FEN
  game = new Chess(fen);

  // Update FEN display
  fenDisplay.value = fen;

  // Update move status
  updateMoveStatus();

  // Create chessboard.js board
  const config = {
    draggable: true,
    position: fen,
    orientation: playerColor === 'w' ? 'white' : 'black',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd,
    pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
  };

  board = Chessboard('chess-board', config);

  // Resize board responsively
  window.addEventListener('resize', () => board && board.resize());
}

/* ------------------------------------------------------------------ */
/* chess.js move-validation callbacks                                   */
/* ------------------------------------------------------------------ */

/** Prevent picking up pieces when the game is over or it's not this player's turn. */
function onDragStart(source, piece) {
  if (game.game_over()) return false;
  // Only allow the current player's pieces to be dragged
  if (game.turn() === 'w' && piece.startsWith('b')) return false;
  if (game.turn() === 'b' && piece.startsWith('w')) return false;
  return true;
}

/** Validate the attempted move; snap back if illegal. */
function onDrop(source, target) {
  const move = game.move({
    from: source,
    to: target,
    promotion: 'q', // Auto-promote to queen
  });

  if (move === null) return 'snapback';

  // Update FEN display after a legal move
  fenDisplay.value = game.fen();

  updateMoveStatus();
}

/** After the piece snaps into place, sync the board with the game state. */
function onSnapEnd() {
  board.position(game.fen());
}

/* ------------------------------------------------------------------ */
/* Game-state helpers                                                   */
/* ------------------------------------------------------------------ */

function updateMoveStatus() {
  if (!game) return;

  if (game.in_checkmate()) {
    const winner = game.turn() === 'w' ? 'Black' : 'White';
    moveStatus.textContent = `♛ Checkmate! ${winner} wins.`;
  } else if (game.in_stalemate()) {
    moveStatus.textContent = '⚖️ Stalemate! The game is a draw.';
  } else if (game.in_draw()) {
    moveStatus.textContent = '⚖️ Draw!';
  } else if (game.in_check()) {
    const inCheck = game.turn() === 'w' ? 'White' : 'Black';
    moveStatus.textContent = `⚠️ ${inCheck} is in check!`;
  } else {
    const toMove = game.turn() === 'w' ? 'White' : 'Black';
    moveStatus.textContent = `${toMove} to move`;
  }
}

/* ------------------------------------------------------------------ */
/* Control button handlers                                              */
/* ------------------------------------------------------------------ */

/** Play as White — re-orient the board. */
playWhiteBtn.addEventListener('click', () => {
  playerColor = 'w';
  playWhiteBtn.classList.add('active');
  playBlackBtn.classList.remove('active');
  if (board) board.orientation('white');
});

/** Play as Black — flip the board. */
playBlackBtn.addEventListener('click', () => {
  playerColor = 'b';
  playBlackBtn.classList.add('active');
  playWhiteBtn.classList.remove('active');
  if (board) board.orientation('black');
});

/** Reset to the originally detected position. */
resetBtn.addEventListener('click', () => {
  if (detectedFen) {
    initChessBoard(detectedFen);
  }
});

/** Go back to the upload view. */
newUploadBtn.addEventListener('click', () => {
  resultCard.hidden = true;
  selectedFile = null;
  previewContainer.hidden = true;
  previewImg.src = '';
  detectBtn.disabled = true;
  fileInput.value = '';
  showStatus('', '');
  uploadCard.scrollIntoView({ behavior: 'smooth' });
});

/** Copy the FEN to the clipboard. */
copyFenBtn.addEventListener('click', async () => {
  if (!fenDisplay.value) return;
  try {
    await navigator.clipboard.writeText(fenDisplay.value);
    const original = copyFenBtn.textContent;
    copyFenBtn.textContent = '✓ Copied!';
    setTimeout(() => { copyFenBtn.textContent = original; }, 2000);
  } catch {
    // Fallback for older browsers
    fenDisplay.select();
    document.execCommand('copy');
  }
});
