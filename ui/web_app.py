import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from flask import Flask, render_template, request, jsonify
import chess
from engine.board import EngineBoard
from engine.engine import ChessEngine
from engine.search import negamax, SearchInfo, INF, MATE_SCORE
from engine.transposition import TranspositionTable

app = Flask(__name__)

# Pre-load both engines at startup so toggling is instant
game_board = EngineBoard()
engine = ChessEngine(max_depth=64, time_limit=1.0, use_nn=False)
nn_engine = ChessEngine(max_depth=64, time_limit=1.0, use_nn=True)

EVAL_DEPTH = 7  # Search depth for the evaluation bar (7 ply = ~3.5 moves ahead)


def _get_eval_score(board: EngineBoard, use_nn: bool) -> dict:
    """
    Run a search (EVAL_DEPTH plies) and return the score from White's
    perspective in pawn units (e.g. +0.3, -1.5, +2.1).

    Returns a dict: {"score": float, "mate": int|None}
    - "score" is in pawn units (divide centipawns by 100)
    - "mate" is the number of moves to mate (positive=white mates, negative=black mates)
    """
    if board.is_game_over():
        if board.is_checkmate():
            # Side to move is mated
            if board.turn() == chess.WHITE:
                return {"score": -999.0, "mate": 0}
            else:
                return {"score": 999.0, "mate": 0}
        return {"score": 0.0, "mate": None}  # Draw

    active_engine = nn_engine if use_nn else engine
    info = SearchInfo(
        tt=active_engine.tt,
        start_time=time.time(),
        time_limit=2.0  # cap eval computation to 2s
    )
    score, _ = negamax(board, EVAL_DEPTH, -INF, INF, active_engine.eval_fn, info)

    # negamax returns score from side-to-move's perspective.
    # Flip to White's perspective for the eval bar.
    if board.turn() == chess.BLACK:
        score = -score

    # Check for mate scores
    if abs(score) > MATE_SCORE - 1000:
        mate_in = None
        if score > 0:
            mate_in = (MATE_SCORE - abs(score) + 1) // 2
            if mate_in == 0:
                mate_in = 1
        else:
            mate_in = -((MATE_SCORE - abs(score) + 1) // 2)
            if mate_in == 0:
                mate_in = -1
        return {"score": score, "mate": mate_in}

    # Convert centipawn score to pawn units for heuristic eval
    if not use_nn:
        score = score / 100.0

    return {"score": round(score, 2), "mate": None}


def _game_status(board: EngineBoard) -> dict:
    """Return detailed game status info."""
    result = {"game_over": False, "reason": None}
    if board.is_game_over():
        result["game_over"] = True
        if board.is_checkmate():
            # The side to move is checkmated
            winner = "Black" if board.turn() == chess.WHITE else "White"
            result["reason"] = f"Checkmate! {winner} wins."
        elif board.is_stalemate():
            result["reason"] = "Stalemate! Draw."
        else:
            result["reason"] = "Draw."
    elif board.is_check():
        result["reason"] = "Check!"
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/newgame", methods=["POST"])
def new_game():
    global game_board
    game_board = EngineBoard()
    engine.tt.clear()
    nn_engine.tt.clear()
    
    data = request.json or {}
    use_nn = data.get("use_nn", False)
    
    eval_info = _get_eval_score(game_board, use_nn)
    return jsonify({"fen": game_board.fen(), "eval": eval_info, "status": _game_status(game_board)})


@app.route("/switch_engine", methods=["POST"])
def switch_engine():
    """Clear TT when switching evaluator backends."""
    data = request.json
    use_nn = data.get("use_nn", False)
    active_engine = nn_engine if use_nn else engine
    active_engine.tt.clear()
    eval_info = _get_eval_score(game_board, use_nn)
    return jsonify({"ok": True, "eval": eval_info})


@app.route("/move", methods=["POST"])
def make_move():
    global game_board
    # Capture the board locally to prevent race conditions if 'new_game' is called
    board = game_board
    
    data = request.json
    source = data.get("source")
    target = data.get("target")
    use_nn = data.get("use_nn", False)

    # Check if game is already over
    if board.is_game_over():
        return jsonify({"error": "Game is already over", "fen": board.fen(),
                        "status": _game_status(board)})

    # Validate human move against legal moves
    uci_move = f"{source}{target}"

    # Guard against malformed input from chessboard.js (e.g. 'f1offboard')
    if len(uci_move) < 4 or len(uci_move) > 5:
        return jsonify({"error": "Invalid move", "fen": board.fen()})

    try:
        move = chess.Move.from_uci(uci_move)
    except chess.InvalidMoveError:
        return jsonify({"error": "Invalid move", "fen": board.fen()})

    if move not in board.legal_moves():
        # Try promotion to Queen (chessboard.js doesn't have promotion UI)
        try:
            move = chess.Move.from_uci(uci_move + "q")
        except chess.InvalidMoveError:
            return jsonify({"error": "Invalid move", "fen": board.fen()})
        if move not in board.legal_moves():
            return jsonify({"error": "Invalid move", "fen": board.fen()})

    # Apply human move
    board.push(move)

    status = _game_status(board)

    if board.is_game_over():
        eval_info = _get_eval_score(board, use_nn)
        return jsonify({"fen": board.fen(), "eval": eval_info, "status": status,
                        "engine_move": None})

    # Engine makes a move
    active_engine = nn_engine if use_nn else engine
    best_move = active_engine.get_best_move(board)

    if best_move:
        board.push(best_move)

    status = _game_status(board)
    eval_info = _get_eval_score(board, use_nn)

    return jsonify({
        "fen": board.fen(),
        "status": status,
        "engine_move": best_move.uci() if best_move else None,
        "eval": eval_info
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
