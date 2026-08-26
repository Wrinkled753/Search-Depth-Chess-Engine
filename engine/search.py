import chess
from engine.board import EngineBoard
from engine.evaluate import evaluate

# Infinity values for alpha-beta pruning
INF = float('inf')

def minimax(board: EngineBoard, depth: int, alpha: float, beta: float, maximizing_player: bool) -> tuple[float, chess.Move | None]:
    """
    Minimax search with alpha-beta pruning.
    Returns a tuple of (best_score, best_move).
    """
    if depth == 0 or board.is_game_over():
        return evaluate(board), None

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
            eval_score, _ = minimax(board, depth - 1, alpha, beta, False)
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
            eval_score, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                
            beta = min(beta, eval_score)
            if beta <= alpha:
                break # Alpha cutoff
                
        return min_eval, best_move
