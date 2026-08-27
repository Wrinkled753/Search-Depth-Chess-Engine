import chess
from typing import Callable
from engine.board import EngineBoard

# Infinity values for alpha-beta pruning
INF = float('inf')
MATE_SCORE = 99999.0

def minimax(board: EngineBoard, depth: int, alpha: float, beta: float, maximizing_player: bool, eval_fn: Callable[[EngineBoard], float]) -> tuple[float, chess.Move | None]:
    """
    Minimax search with alpha-beta pruning.
    
    The eval_fn parameter allows swapping evaluation backends
    (heuristic vs neural network) without changing the search logic.
    
    Terminal state detection (checkmate, stalemate) is handled here,
    so eval_fn only needs to score non-terminal positions.
    
    Returns a tuple of (best_score, best_move).
    """
    # Terminal state checks — shared for ALL eval backends
    if board.is_checkmate():
        return (-MATE_SCORE if board.turn() == chess.WHITE else MATE_SCORE), None
    if board.is_stalemate() or board.is_game_over():
        return 0.0, None
    
    # Leaf node — delegate to the injected evaluation function
    if depth == 0:
        return eval_fn(board), None

    best_move = None
    
    # Simple move ordering: captures first, then promotions, then quiet moves
    # This significantly improves alpha-beta pruning efficiency
    legal_moves = list(board.legal_moves())
    
    def move_score(move: chess.Move) -> int:
        score = 0
        if board._board.is_capture(move):
            score += 10
        if move.promotion:
            score += 5
        # If it's a check, we could also score it higher, etc.
        return score

    legal_moves.sort(key=move_score, reverse=True)

    if maximizing_player:
        max_eval = -INF
        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, False, eval_fn)
            board.pop()
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break # Beta cutoff
                
        return max_eval, best_move
        
    else:
        min_eval = INF
        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, True, eval_fn)
            board.pop()
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                
            beta = min(beta, eval_score)
            if beta <= alpha:
                break # Alpha cutoff
                
        return min_eval, best_move
