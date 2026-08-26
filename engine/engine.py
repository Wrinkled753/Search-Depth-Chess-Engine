import chess
from engine.board import EngineBoard
from engine.search import minimax

class ChessEngine:
    """
    The main Chess Engine class that binds the board, evaluation, and search logic.
    """
    
    def __init__(self, depth: int = 4):
        self.depth = depth

    def get_best_move(self, board: EngineBoard) -> chess.Move | None:
        """
        Finds the best move for the current position using the configured search depth.
        """
        maximizing = (board.turn() == chess.WHITE)
        
        score, best_move = minimax(
            board, 
            depth=self.depth, 
            alpha=float('-inf'), 
            beta=float('inf'), 
            maximizing_player=maximizing
        )
        
        # If no legal moves or best move could be determined, fallback to first legal move
        # (Usually only happens if checkmate/stalemate logic failed to catch an end state 
        # before calling get_best_move, or depth is 0)
        if best_move is None and not board.is_game_over():
            try:
                best_move = next(board.legal_moves())
            except StopIteration:
                pass

        return best_move
