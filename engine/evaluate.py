import chess
from engine.board import EngineBoard

# Piece values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0  # King value is not used in standard material counting
}

# Center squares for small mobility/center control bonus
CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
EXTENDED_CENTER = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4,
    chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6
}

def evaluate(board: EngineBoard) -> float:
    """
    Evaluates the current board position.
    Returns a positive score if white is better, negative if black is better.
    """
    if board.is_checkmate():
        # If it's checkmate, the side to move lost.
        return -99999.0 if board.turn() == chess.WHITE else 99999.0
    
    if board.is_stalemate() or board.is_game_over():
        return 0.0

    score = 0.0
    b = board._board  # Access the underlying python-chess board
    
    # Material evaluation & Center control
    for square in chess.SQUARES:
        piece = b.piece_at(square)
        if piece is not None:
            # Material
            value = PIECE_VALUES.get(piece.piece_type, 0)
            
            # Center bonus
            center_bonus = 0.0
            if piece.piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP):
                if square in CENTER_SQUARES:
                    center_bonus = 20.0
                elif square in EXTENDED_CENTER:
                    center_bonus = 10.0

            # King safety penalty (very basic: penalized if king is in the center)
            king_safety = 0.0
            if piece.piece_type == chess.KING:
                if square in CENTER_SQUARES or square in EXTENDED_CENTER:
                    king_safety = -30.0

            total_piece_value = value + center_bonus + king_safety
            
            if piece.color == chess.WHITE:
                score += total_piece_value
            else:
                score -= total_piece_value

    return score
