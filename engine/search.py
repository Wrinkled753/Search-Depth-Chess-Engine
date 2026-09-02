import time
import chess
import chess.polyglot
from typing import Callable
from dataclasses import dataclass, field
from engine.board import EngineBoard
from engine.transposition import TranspositionTable

# Constants
INF = 1000000.0
MATE_SCORE = 99999.0

# Piece values for MVV-LVA move ordering and Delta Pruning
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
    
    # Killer moves: [ply][0 or 1]
    killer_moves: list[list[chess.Move | None]] = field(default_factory=lambda: [[None, None] for _ in range(128)])
    # History heuristic table [color][from_sq][to_sq]
    history_table: list[list[list[int]]] = field(
        default_factory=lambda: [[[0 for _ in range(64)] for _ in range(64)] for _ in range(2)]
    )
    
    def check_time(self) -> None:
        """Check if the time limit has been exceeded."""
        if self.time_limit > 0 and time.time() - self.start_time >= self.time_limit:
            self.stopped = True


def _side_relative_eval(board: EngineBoard, eval_fn: Callable) -> float:
    """
    Wraps eval_fn to guarantee side-to-move relative output.
    """
    raw = eval_fn(board)
    if board.turn() == chess.BLACK:
        return -raw
    return raw


def _mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """
    Most Valuable Victim - Least Valuable Attacker move ordering.
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


def _order_moves(board: EngineBoard, moves: list[chess.Move], tt_move: chess.Move | None, search_info: SearchInfo, ply: int) -> list[chess.Move]:
    """
    Order moves for better alpha-beta pruning efficiency.
    Priority: TT best move > Captures (MVV-LVA) > Killer Moves > History Heuristic.
    """
    color = 1 if board.turn() == chess.WHITE else 0
    
    def sort_key(move: chess.Move) -> int:
        # TT move gets highest priority
        if tt_move and move == tt_move:
            return 10000000
            
        if board._board.is_capture(move):
            return 1000000 + _mvv_lva_score(board._board, move)
            
        # Killer moves
        if ply < 128:
            if move == search_info.killer_moves[ply][0]:
                return 900000
            if move == search_info.killer_moves[ply][1]:
                return 800000
                
        # History heuristic
        return search_info.history_table[color][move.from_square][move.to_square]
    
    moves.sort(key=sort_key, reverse=True)
    return moves


def quiescence(board: EngineBoard, alpha: float, beta: float,
               eval_fn: Callable, search_info: SearchInfo) -> float:
    """
    Quiescence search to resolve tactical sequences at leaf nodes.
    Includes Delta Pruning.
    """
    search_info.nodes_searched += 1
    
    if search_info.stopped:
        return 0.0
    
    if search_info.nodes_searched % 4096 == 0:
        search_info.check_time()
        if search_info.stopped:
            return 0.0
    
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_game_over():
        return 0.0
    
    in_check = board.is_check()
    stand_pat = -INF
    
    if not in_check:
        stand_pat = _side_relative_eval(board, eval_fn)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
    
    if in_check:
        moves = list(board.legal_moves())
    else:
        moves = list(board._board.generate_legal_captures())
    
    # Sort captures by MVV-LVA
    moves.sort(key=lambda m: _mvv_lva_score(board._board, m), reverse=True)
    
    for move in moves:
        # Delta Pruning
        if not in_check and not move.promotion:
            captured_piece = board._board.piece_at(move.to_square)
            if captured_piece:
                captured_val = PIECE_VALUES.get(captured_piece.piece_type, 0)
                # If stand_pat + captured piece value + 200 centipawns margin < alpha,
                # this capture is unlikely to improve the position enough.
                if stand_pat + captured_val + 200 < alpha:
                    continue
                    
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
            eval_fn: Callable, search_info: SearchInfo, ply: int = 0, is_null_move: bool = False) -> tuple[float, chess.Move | None]:
    """
    Negamax search with Alpha-Beta pruning, TT, QSearch, Move Ordering (Killer+History),
    Null Move Pruning (NMP), and Late Move Reduction (LMR).
    """
    search_info.nodes_searched += 1
    
    if search_info.stopped:
        return 0.0, None
    
    if search_info.nodes_searched % 4096 == 0:
        search_info.check_time()
        if search_info.stopped:
            return 0.0, None
    
    if board.is_checkmate():
        # Prefer faster mates
        return -MATE_SCORE + ply, None
    if board.is_stalemate() or board.is_game_over():
        return 0.0, None
    
    if depth <= 0:
        return quiescence(board, alpha, beta, eval_fn, search_info), None
    
    in_check = board.is_check()
    if in_check:
        # Check extension
        depth += 1
    
    zobrist_key = board.zobrist_hash()
    hit, tt_score, tt_move = search_info.tt.probe(zobrist_key, depth, alpha, beta)
    if hit and ply > 0: # Don't prune at root to ensure we return a move
        return tt_score, tt_move
        
    # Null Move Pruning (NMP)
    if not in_check and not is_null_move and depth >= 3 and ply > 0:
        # A simple check to avoid zugzwang: only if we have pieces other than pawns and king
        # For simplicity, we omit the material check here, but normally you'd check non-pawn material.
        R = 2
        board.push_null()
        null_score, _ = negamax(board, depth - 1 - R, -beta, -beta + 1, eval_fn, search_info, ply + 1, is_null_move=True)
        null_score = -null_score
        board.pop_null()
        
        if search_info.stopped:
            return 0.0, None
            
        if null_score >= beta:
            return beta, None

    legal_moves = list(board.legal_moves())
    legal_moves = _order_moves(board, legal_moves, tt_move, search_info, ply)
    
    best_move = None
    best_score = -INF
    original_alpha = alpha
    
    for move_count, move in enumerate(legal_moves):
        is_capture = board._board.is_capture(move)
        board.push(move)
        
        # Late Move Reduction (LMR)
        # Apply LMR for quiet moves later in the ordering, provided we aren't in check.
        needs_full_search = True
        if depth >= 3 and move_count >= 3 and not in_check and not is_capture and not move.promotion:
            reduction = 1
            if move_count >= 6:
                reduction = 2
            
            # Reduced depth search
            score, _ = negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, eval_fn, search_info, ply + 1, is_null_move)
            score = -score
            
            if score <= alpha:
                needs_full_search = False
            else:
                needs_full_search = True

        if needs_full_search:
            # Principal Variation Search (PVS)
            if move_count == 0:
                score, _ = negamax(board, depth - 1, -beta, -alpha, eval_fn, search_info, ply + 1, is_null_move)
                score = -score
            else:
                score, _ = negamax(board, depth - 1, -alpha - 1, -alpha, eval_fn, search_info, ply + 1, is_null_move)
                score = -score
                if alpha < score < beta:
                    score, _ = negamax(board, depth - 1, -beta, -alpha, eval_fn, search_info, ply + 1, is_null_move)
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
            # Beta Cutoff
            if not is_capture:
                # Update Killer Moves
                if ply < 128 and search_info.killer_moves[ply][0] != move:
                    search_info.killer_moves[ply][1] = search_info.killer_moves[ply][0]
                    search_info.killer_moves[ply][0] = move
                
                # Update History Heuristic
                color = 1 if board.turn() == chess.WHITE else 0
                search_info.history_table[color][move.from_square][move.to_square] += depth * depth
            break
            
    if not search_info.stopped:
        if best_score <= original_alpha:
            flag = TranspositionTable.ALPHA
        elif best_score >= beta:
            flag = TranspositionTable.BETA
        else:
            flag = TranspositionTable.EXACT
        search_info.tt.store(zobrist_key, depth, best_score, flag, best_move)
    
    return best_score, best_move
