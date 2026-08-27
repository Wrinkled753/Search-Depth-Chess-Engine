import time
import chess
import chess.polyglot
from typing import Callable
from dataclasses import dataclass, field
from engine.board import EngineBoard
from engine.transposition import TranspositionTable

# Constants
INF = float('inf')
MATE_SCORE = 99999.0

# Piece values for MVV-LVA move ordering
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

@dataclass
class SearchInfo:
    """Carries shared mutable state through the recursive search."""
    nodes_searched: int = 0
    tt: TranspositionTable = field(default_factory=TranspositionTable)
    start_time: float = 0.0
    time_limit: float = 5.0
    stopped: bool = False
    
    def check_time(self) -> None:
        """Check if the time limit has been exceeded."""
        if self.time_limit > 0 and time.time() - self.start_time >= self.time_limit:
            self.stopped = True


def _side_relative_eval(board: EngineBoard, eval_fn: Callable) -> float:
    """
    Wraps eval_fn to guarantee side-to-move relative output.
    Our eval functions return positive=white advantage.
    Negamax requires positive=good for the side to move.
    """
    raw = eval_fn(board)
    if board.turn() == chess.BLACK:
        return -raw
    return raw


def _mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """
    Most Valuable Victim - Least Valuable Attacker move ordering.
    Higher scores indicate better captures to try first.
    Example: Queen captured by Pawn = 900 - 100 = 800 (excellent)
    """
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score = PIECE_VALUES.get(victim.piece_type, 0) - PIECE_VALUES.get(attacker.piece_type, 0)
    if move.promotion:
        score += 800
    return score


def _order_moves(board: EngineBoard, moves: list[chess.Move], tt_move: chess.Move | None) -> list[chess.Move]:
    """
    Order moves for better alpha-beta pruning efficiency.
    Priority: TT best move > captures (MVV-LVA) > promotions > quiet moves.
    """
    def sort_key(move: chess.Move) -> int:
        # TT move gets highest priority
        if tt_move and move == tt_move:
            return 100000
        return _mvv_lva_score(board._board, move)
    
    moves.sort(key=sort_key, reverse=True)
    return moves


def quiescence(board: EngineBoard, alpha: float, beta: float,
               eval_fn: Callable, search_info: SearchInfo) -> float:
    """
    Quiescence search to resolve tactical sequences at leaf nodes.
    Prevents the horizon effect by continuing to search captures
    (and all legal moves when in check) beyond the main search depth.
    """
    search_info.nodes_searched += 1
    
    if search_info.stopped:
        return 0.0
    
    # Check time periodically (every 4096 nodes)
    if search_info.nodes_searched % 4096 == 0:
        search_info.check_time()
        if search_info.stopped:
            return 0.0
    
    # Terminal state checks
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_game_over():
        return 0.0
    
    in_check = board.is_check()
    
    # Stand-pat: if not in check, use static eval as a lower bound
    if not in_check:
        stand_pat = _side_relative_eval(board, eval_fn)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
    
    # In check: search ALL legal moves (must escape check)
    # Not in check: search only captures and promotions
    if in_check:
        moves = list(board.legal_moves())
    else:
        moves = list(board._board.generate_legal_captures())
    
    for move in moves:
        board.push(move)
        score = -quiescence(board, -beta, -alpha, eval_fn, search_info)
        board.pop()
        
        if search_info.stopped:
            return 0.0
        
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    
    return alpha


def negamax(board: EngineBoard, depth: int, alpha: float, beta: float,
            eval_fn: Callable, search_info: SearchInfo) -> tuple[float, chess.Move | None]:
    """
    Negamax search with alpha-beta pruning, transposition table,
    quiescence search, and MVV-LVA move ordering.
    
    Score is always from the perspective of the side to move
    (positive = good for the side to move).
    
    Returns a tuple of (best_score, best_move).
    """
    search_info.nodes_searched += 1
    
    if search_info.stopped:
        return 0.0, None
    
    # Check time periodically
    if search_info.nodes_searched % 4096 == 0:
        search_info.check_time()
        if search_info.stopped:
            return 0.0, None
    
    # Terminal state checks
    if board.is_checkmate():
        return -MATE_SCORE, None
    if board.is_stalemate() or board.is_game_over():
        return 0.0, None
    
    # Leaf node — enter quiescence search
    if depth <= 0:
        return quiescence(board, alpha, beta, eval_fn, search_info), None
    
    # Transposition table probe
    tt_move = None
    zobrist_key = board.zobrist_hash()
    hit, tt_score, tt_move = search_info.tt.probe(zobrist_key, depth, alpha, beta)
    if hit:
        return tt_score, tt_move
    
    # Generate and order moves
    legal_moves = list(board.legal_moves())
    legal_moves = _order_moves(board, legal_moves, tt_move)
    
    best_move = None
    best_score = -INF
    original_alpha = alpha
    
    for move in legal_moves:
        board.push(move)
        score, _ = negamax(board, depth - 1, -beta, -alpha, eval_fn, search_info)
        score = -score
        board.pop()
        
        if search_info.stopped:
            return 0.0, None
        
        if score > best_score:
            best_score = score
            best_move = move
        
        if score > alpha:
            alpha = score
        
        if alpha >= beta:
            break  # Beta cutoff
    
    # Store result in transposition table
    if not search_info.stopped:
        if best_score <= original_alpha:
            flag = TranspositionTable.ALPHA
        elif best_score >= beta:
            flag = TranspositionTable.BETA
        else:
            flag = TranspositionTable.EXACT
        
        search_info.tt.store(zobrist_key, depth, best_score, flag, best_move)
    
    return best_score, best_move
