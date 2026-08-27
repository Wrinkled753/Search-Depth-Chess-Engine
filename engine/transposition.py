class TranspositionTable:
    """
    Fixed-size transposition table using Zobrist hash keys.
    Stores search results to avoid re-evaluating positions that have
    already been analyzed at sufficient depth.
    
    Uses a pre-allocated list with key % max_size indexing to guarantee
    bounded memory usage.
    """
    EXACT = 0   # Exact score
    ALPHA = 1   # Upper bound (failed low)
    BETA = 2    # Lower bound (failed high)
    
    def __init__(self, max_size: int = 2**20):
        self.max_size = max_size
        self.table: list = [None] * max_size
    
    def store(self, key: int, depth: int, score: float, flag: int, best_move) -> None:
        """Store a search result in the transposition table."""
        index = key % self.max_size
        self.table[index] = (key, depth, score, flag, best_move)
    
    def probe(self, key: int, depth: int, alpha: float, beta: float) -> tuple[bool, float, object]:
        """
        Probe the transposition table for a stored result.
        
        Returns (hit, score, tt_move):
        - hit=True means the stored score can be used as a cutoff.
        - tt_move is always returned if available (for move ordering),
          even when hit=False.
        - Only uses stored results when tt_depth >= requested depth.
        """
        index = key % self.max_size
        entry = self.table[index]
        
        if entry is None or entry[0] != key:
            return False, 0.0, None
        
        tt_key, tt_depth, tt_score, tt_flag, tt_move = entry
        
        # Only use the stored score if it was searched to at least
        # the same depth we need now
        if tt_depth < depth:
            # Still return the TT best move for move ordering
            return False, 0.0, tt_move
        
        if tt_flag == self.EXACT:
            return True, tt_score, tt_move
        elif tt_flag == self.ALPHA and tt_score <= alpha:
            return True, alpha, tt_move
        elif tt_flag == self.BETA and tt_score >= beta:
            return True, beta, tt_move
        
        # No cutoff, but still return move for ordering
        return False, 0.0, tt_move
    
    def clear(self) -> None:
        """Clear all entries from the transposition table."""
        self.table = [None] * self.max_size
