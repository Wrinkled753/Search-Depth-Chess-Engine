import chess
from engine.board import EngineBoard
from engine.search import minimax
from engine.evaluate import evaluate as heuristic_eval

class ChessEngine:
    """
    The main Chess Engine class that binds the board, evaluation, and search logic.
    Supports both hand-written heuristic and neural network evaluation backends.
    """
    
    def __init__(self, depth: int = 4, use_nn: bool = False, model_path: str = "models/eval_net.pt"):
        self.depth = depth
        self.use_nn = use_nn
        
        if use_nn:
            from engine.nnue_eval import NNUEEvaluator
            self.eval_fn = NNUEEvaluator(model_path)
        else:
            self.eval_fn = heuristic_eval

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
            maximizing_player=maximizing,
            eval_fn=self.eval_fn
        )
        
        # If no legal moves or best move could be determined, fallback to first legal move
        if best_move is None and not board.is_game_over():
            try:
                best_move = next(board.legal_moves())
            except StopIteration:
                pass

        return best_move
