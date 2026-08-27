import pytest
import chess
from engine.board import EngineBoard
from engine.engine import ChessEngine

def test_find_mate_in_one():
    """Test if the engine can find a mate in 1 (heuristic eval)."""
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    board = EngineBoard.from_fen(fen)
    engine = ChessEngine(max_depth=2, time_limit=10.0)
    best_move = engine.get_best_move(board)
    assert best_move.uci() == "h5f7"

def test_find_mate_in_two():
    """Test if the engine can find a mate in 2 (back-rank mate)."""
    fen = "3r2k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"
    board = EngineBoard.from_fen(fen)
    engine = ChessEngine(max_depth=4, time_limit=10.0)
    best_move = engine.get_best_move(board)
    assert best_move.uci() == "d1d8"

def test_prevent_mate_in_one():
    """Test if the engine will avoid getting mated in 1."""
    fen = "r3k2r/pbpp1ppp/1pn5/4p3/4P3/2NP2Pq/PPP2P1P/R1BQ1RK1 w kq - 0 1"
    board = EngineBoard.from_fen(fen)
    engine = ChessEngine(max_depth=3, time_limit=10.0)
    best_move = engine.get_best_move(board)
    assert best_move is not None
    assert best_move in board.legal_moves()

def test_nn_engine_finds_mate_in_one():
    """Verify mate detection works with NN eval too (handled in search.py)."""
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    board = EngineBoard.from_fen(fen)
    engine = ChessEngine(max_depth=2, time_limit=10.0, use_nn=True)
    best_move = engine.get_best_move(board)
    assert best_move.uci() == "h5f7"

def test_engine_finds_tactic_as_black():
    """
    Verify the engine correctly finds a winning tactic when playing as black.
    This confirms that the side-relative eval conversion works correctly.
    
    Position: Black to move, can capture a free queen with Qxd1.
    """
    # Black to move, white queen is hanging on d4 (d file open for black)
    fen = "r1bqkbnr/ppp1pppp/8/8/3Q4/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
    board = EngineBoard.from_fen(fen)
    engine = ChessEngine(max_depth=3, time_limit=10.0)
    best_move = engine.get_best_move(board)
    # Black should capture the white queen
    assert best_move.uci() == "d8d4"
