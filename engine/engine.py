import time
import chess
from engine.board import EngineBoard
from engine.search import negamax, SearchInfo, INF
from engine.transposition import TranspositionTable
from engine.evaluate import evaluate as heuristic_eval

class ChessEngine:
    """
    The main Chess Engine class that binds the board, evaluation, and search logic.
    Supports both hand-written heuristic and neural network evaluation backends.
    Uses iterative deepening with a time limit for search.
    """
    
    def __init__(self, max_depth: int = 64, time_limit: float = 5.0,
                 use_nn: bool = False, model_path: str = "models/eval_net.pt"):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.use_nn = use_nn
        self.tt = TranspositionTable()
        
        if use_nn:
            from engine.nnue_eval import NNUEEvaluator
            self.eval_fn = NNUEEvaluator(model_path)
        else:
            self.eval_fn = heuristic_eval

    def get_best_move(self, board: EngineBoard, time_limit: float | None = None) -> tuple[chess.Move | None, dict]:
        """
        Finds the best move using iterative deepening.
        Searches progressively deeper until the time limit is reached.
        If a search iteration is interrupted, its result is discarded
        and the last fully completed iteration's move is used.
        
        Returns a tuple of (best_move, search_stats).
        search_stats contains: depth, nodes, time_s, nps
        """
        limit = time_limit if time_limit is not None else self.time_limit
        
        search_info = SearchInfo(
            tt=self.tt,
            start_time=time.time(),
            time_limit=limit
        )
        
        best_move = None
        best_score = -INF
        completed_depth = 0
        total_nodes = 0
        
        for depth in range(1, self.max_depth + 1):
            search_info.stopped = False
            search_info.nodes_searched = 0
            
            score, move = negamax(
                board, depth, -INF, INF,
                eval_fn=self.eval_fn,
                search_info=search_info
            )
            
            total_nodes += search_info.nodes_searched
            
            # If search was interrupted mid-iteration, discard this
            # incomplete result and use the previous completed depth.
            if search_info.stopped:
                break
            
            completed_depth = depth
            if move is not None:
                best_move = move
                best_score = score
        
        elapsed = time.time() - search_info.start_time
        nps = int(total_nodes / elapsed) if elapsed > 0 else 0
        
        search_stats = {
            "depth": completed_depth,
            "nodes": total_nodes,
            "time_s": round(elapsed, 2),
            "nps": nps,
        }
        
        # Fallback: if no move found (shouldn't happen in normal play)
        if best_move is None and not board.is_game_over():
            try:
                best_move = next(iter(board.legal_moves()))
            except StopIteration:
                pass

        return best_move, search_stats
