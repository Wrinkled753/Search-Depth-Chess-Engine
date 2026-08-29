import sys
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from flask import Flask, render_template, request, jsonify
import chess
from engine.board import EngineBoard
from engine.engine import ChessEngine
from engine.evaluate import evaluate as heuristic_eval

app = Flask(__name__)

# Pre-load both engines at startup so toggling is instant
game_board = EngineBoard()
engine = ChessEngine(max_depth=64, time_limit=1.0, use_nn=False)
nn_engine = ChessEngine(max_depth=64, time_limit=1.0, use_nn=True)

def _get_eval_score(board: EngineBoard, use_nn: bool) -> float:
    """
    Return the evaluation score in centipawns from White's perspective.
    Positive = White advantage, Negative = Black advantage.
    """
    active_engine = nn_engine if use_nn else engine
    raw = active_engine.eval_fn(board)
    return round(raw, 2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/newgame", methods=["POST"])
def new_game():
    global game_board
    game_board = EngineBoard()
    # Clear TT for both engines on new game
    engine.tt.clear()
    nn_engine.tt.clear()
    eval_score = _get_eval_score(game_board, False)
    return jsonify({"fen": game_board.fen(), "eval": eval_score})

@app.route("/switch_engine", methods=["POST"])
def switch_engine():
    """Clear TT when switching evaluator backends."""
    data = request.json
    use_nn = data.get("use_nn", False)
    active_engine = nn_engine if use_nn else engine
    active_engine.tt.clear()
    eval_score = _get_eval_score(game_board, use_nn)
    return jsonify({"ok": True, "eval": eval_score})

@app.route("/move", methods=["POST"])
def make_move():
    global game_board
    data = request.json
    source = data.get("source")
    target = data.get("target")
    use_nn = data.get("use_nn", False)
    
    # Check if game is over
    if game_board.is_game_over():
        return jsonify({"error": "Game is already over", "fen": game_board.fen()})

    # Validate human move against legal moves
    uci_move = f"{source}{target}"
    
    # Guard against malformed input from chessboard.js (e.g. 'f1offboard')
    if len(uci_move) < 4 or len(uci_move) > 5:
        return jsonify({"error": "Invalid move", "fen": game_board.fen()})
    
    try:
        move = chess.Move.from_uci(uci_move)
    except chess.InvalidMoveError:
        return jsonify({"error": "Invalid move", "fen": game_board.fen()})
    
    if move not in game_board.legal_moves():
        # Try promotion to Queen (chessboard.js doesn't have promotion UI)
        try:
            move = chess.Move.from_uci(uci_move + "q")
        except chess.InvalidMoveError:
            return jsonify({"error": "Invalid move", "fen": game_board.fen()})
        if move not in game_board.legal_moves():
            return jsonify({"error": "Invalid move", "fen": game_board.fen()})
    
    # Apply human move
    game_board.push(move)
    
    if game_board.is_game_over():
        eval_score = _get_eval_score(game_board, use_nn)
        return jsonify({"fen": game_board.fen(), "game_over": True, "eval": eval_score})
        
    # Engine makes a move
    active_engine = nn_engine if use_nn else engine
    best_move = active_engine.get_best_move(game_board)
    
    if best_move:
        game_board.push(best_move)
    
    eval_score = _get_eval_score(game_board, use_nn)
        
    return jsonify({
        "fen": game_board.fen(),
        "game_over": game_board.is_game_over(),
        "engine_move": best_move.uci() if best_move else None,
        "eval": eval_score
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
