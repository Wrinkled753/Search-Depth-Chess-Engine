"""
Benchmark script comparing old (plain alpha-beta) vs new (optimized) search.
Measures nodes searched and depth reached within a time limit.
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import chess
from engine.board import EngineBoard
from engine.engine import ChessEngine
from engine.search import negamax, SearchInfo, INF
from engine.transposition import TranspositionTable
from engine.evaluate import evaluate as heuristic_eval

# Test positions for benchmarking
BENCH_POSITIONS = [
    ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("Middlegame", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7"),
    ("Complex Tactics", "r2qkb1r/pp2pppp/2n2n2/3p1b2/3P4/2N2N2/PPP1BPPP/R1BQK2R w KQkq - 4 6"),
]

def benchmark_fixed_depth():
    """Compare nodes searched at fixed depths."""
    print("=" * 70)
    print("  BENCHMARK: Nodes Searched at Fixed Depth (Optimized Search)")
    print("=" * 70)
    print(f"{'Position':<22} {'Depth':<7} {'Nodes':<12} {'Time (s)':<10}")
    print("-" * 51)
    
    for name, fen in BENCH_POSITIONS:
        board = EngineBoard.from_fen(fen)
        for depth in [3, 4, 5]:
            search_info = SearchInfo(
                tt=TranspositionTable(),
                start_time=time.time(),
                time_limit=0  # No time limit for fixed depth
            )
            
            start = time.time()
            score, move = negamax(board, depth, -INF, INF, heuristic_eval, search_info)
            elapsed = time.time() - start
            
            print(f"{name:<22} {depth:<7} {search_info.nodes_searched:<12} {elapsed:<10.3f}")
    print()

def benchmark_iterative_deepening():
    """Measure max depth reached within time limits."""
    print("=" * 70)
    print("  BENCHMARK: Iterative Deepening (Max Depth in Time Limit)")
    print("=" * 70)
    print(f"{'Position':<22} {'Time Limit':<12} {'Depth':<7} {'Nodes':<12} {'Move':<8}")
    print("-" * 61)
    
    for name, fen in BENCH_POSITIONS:
        board = EngineBoard.from_fen(fen)
        for limit in [1.0, 3.0, 5.0]:
            engine = ChessEngine(max_depth=64, time_limit=limit)
            
            # Track depth by running iterative deepening manually
            tt = TranspositionTable()
            best_move = None
            max_depth_reached = 0
            total_nodes = 0
            
            for depth in range(1, 64):
                search_info = SearchInfo(
                    tt=tt,
                    start_time=time.time() if depth == 1 else search_info.start_time,
                    time_limit=limit
                )
                if depth == 1:
                    start_time = time.time()
                    search_info.start_time = start_time
                
                score, move = negamax(board, depth, -INF, INF, heuristic_eval, search_info)
                
                if search_info.stopped:
                    break
                
                if move:
                    best_move = move
                max_depth_reached = depth
                total_nodes += search_info.nodes_searched
            
            move_str = best_move.uci() if best_move else "None"
            print(f"{name:<22} {limit:<12.1f} {max_depth_reached:<7} {total_nodes:<12} {move_str:<8}")
    print()

def benchmark_tt_effectiveness():
    """Show TT hit rate by running the same position twice."""
    print("=" * 70)
    print("  BENCHMARK: Transposition Table Effectiveness")
    print("=" * 70)
    
    name, fen = BENCH_POSITIONS[2]  # Middlegame position
    board = EngineBoard.from_fen(fen)
    depth = 4
    
    # First run: cold TT
    tt = TranspositionTable()
    info1 = SearchInfo(tt=tt, start_time=time.time(), time_limit=0)
    start1 = time.time()
    negamax(board, depth, -INF, INF, heuristic_eval, info1)
    time1 = time.time() - start1
    
    # Second run: warm TT (from iterative deepening simulation)
    info2 = SearchInfo(tt=tt, start_time=time.time(), time_limit=0)
    start2 = time.time()
    negamax(board, depth, -INF, INF, heuristic_eval, info2)
    time2 = time.time() - start2
    
    print(f"Position: {name} at depth {depth}")
    print(f"  Cold TT: {info1.nodes_searched:>8} nodes, {time1:.3f}s")
    print(f"  Warm TT: {info2.nodes_searched:>8} nodes, {time2:.3f}s")
    improvement = (1 - info2.nodes_searched / max(info1.nodes_searched, 1)) * 100
    print(f"  Node reduction: {improvement:.1f}%")
    print()

if __name__ == "__main__":
    benchmark_fixed_depth()
    benchmark_iterative_deepening()
    benchmark_tt_effectiveness()
