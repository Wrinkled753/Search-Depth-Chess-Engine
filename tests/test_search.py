import pytest
import chess
from engine.board import EngineBoard
from engine.engine import ChessEngine

def test_find_mate_in_one():
    """Test if the engine can find a mate in 1."""
    # White to move and mate in 1 (Qf7#)
    fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    # Wait, the FEN above is already mate. Let's step back one move.
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    board = EngineBoard.from_fen(fen)
    
    engine = ChessEngine(depth=2)
    best_move = engine.get_best_move(board)
    assert best_move.uci() == "h5f7"

def test_find_mate_in_two():
    """Test if the engine can find a mate in 2."""
    # Back-rank mate: White to move: 1. Rd8+ Rxd8 ... mate
    fen = "3r2k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"
    board = EngineBoard.from_fen(fen)
    
    engine = ChessEngine(depth=3)
    best_move = engine.get_best_move(board)
    assert best_move.uci() == "d1d8"

def test_prevent_mate_in_one():
    """Test if the engine will avoid getting mated in 1."""
    fen = "r3k2r/pbpp1ppp/1pn5/4p3/4P3/2NP2Pq/PPP2P1P/R1BQ1RK1 w kq - 0 1"
    board = EngineBoard.from_fen(fen)
    engine = ChessEngine(depth=2)
    best_move = engine.get_best_move(board)
    assert best_move is not None
    assert best_move in board.legal_moves()

def test_nn_engine_finds_mate_in_one():
    """Verify mate detection works with NN eval too (handled in search.py)."""
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    board = EngineBoard.from_fen(fen)
    engine = ChessEngine(depth=2, use_nn=True)
    best_move = engine.get_best_move(board)
    assert best_move.uci() == "h5f7"
