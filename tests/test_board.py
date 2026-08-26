import pytest
import chess
from engine.board import EngineBoard

def test_initial_board_state():
    """Test that an uninitialized board starts in the standard position."""
    board = EngineBoard()
    assert board.to_fen() == chess.STARTING_FEN

def test_from_fen_and_to_fen():
    """Test loading and exporting FEN strings."""
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    board = EngineBoard.from_fen(fen)
    assert board.to_fen() == fen

def test_from_pgn_and_to_pgn():
    """Test loading from PGN and exporting to PGN."""
    pgn_str = "1. e4 e5 2. Nf3 Nc6"
    board = EngineBoard.from_pgn(pgn_str)
    # Check if the board reaches the expected FEN
    expected_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    assert board.to_fen() == expected_fen
    # Exporting should yield the same move sequence
    out_pgn = board.to_pgn()
    assert "1. e4 e5 2. Nf3 Nc6" in out_pgn

def test_generate_legal_moves():
    """Test that the initial board has exactly 20 legal moves."""
    board = EngineBoard()
    moves = list(board.legal_moves())
    assert len(moves) == 20

def test_push_and_pop_move():
    """Test making a move and undoing it."""
    board = EngineBoard()
    initial_fen = board.to_fen()
    
    move = chess.Move.from_uci("e2e4")
    board.push(move)
    assert board.to_fen() != initial_fen
    
    popped = board.pop()
    assert popped == move
    assert board.to_fen() == initial_fen

def test_push_uci():
    """Test making a move via UCI string."""
    board = EngineBoard()
    board.push_uci("e2e4")
    assert "e4" in board.to_pgn()
    
    with pytest.raises(ValueError):
        # Illegal move
        board.push_uci("e2e5")

def test_castling_legality():
    """Test that castling moves are generated legally."""
    # FEN with clear path for white short and long castling
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    board = EngineBoard.from_fen(fen)
    moves = [m.uci() for m in board.legal_moves()]
    assert "e1g1" in moves # Kingside castling
    assert "e1c1" in moves # Queenside castling

def test_en_passant():
    """Test that en passant moves are generated when applicable."""
    # FEN where black just moved pawn from d7 to d5, allowing white e5 pawn to capture d6
    fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    board = EngineBoard.from_fen(fen)
    moves = [m.uci() for m in board.legal_moves()]
    assert "e5d6" in moves

def test_pawn_promotion():
    """Test pawn promotion moves."""
    # White pawn on h7, about to promote
    fen = "8/7P/8/8/8/8/8/k6K w - - 0 1"
    board = EngineBoard.from_fen(fen)
    moves = [m.uci() for m in board.legal_moves()]
    # Should include various promotions: Queen (q), Rook (r), Bishop (b), Knight (n)
    assert "h7h8q" in moves
    assert "h7h8r" in moves
    assert "h7h8b" in moves
    assert "h7h8n" in moves

def test_check_state():
    """Test if board correctly identifies check state."""
    # FEN where black king is in check from white rook
    fen = "4k3/4R3/8/8/8/8/8/7K b - - 0 1"
    board = EngineBoard.from_fen(fen)
    assert board.is_check() is True
    assert board.is_checkmate() is False

def test_checkmate_state():
    """Test if board correctly identifies checkmate state."""
    # Fool's mate FEN
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    board = EngineBoard.from_fen(fen)
    assert board.is_checkmate() is True
    assert board.is_game_over() is True
    assert board.is_check() is True

def test_stalemate_state():
    """Test if board correctly identifies stalemate state."""
    fen = "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"
    board = EngineBoard.from_fen(fen)
    assert board.is_stalemate() is True
    assert board.is_game_over() is True
    assert board.is_checkmate() is False
