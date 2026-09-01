import sys
from pathlib import Path
import time
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from engine.board import EngineBoard
from engine.engine import ChessEngine

def run_benchmark():
    # Middlegame position with lots of tactical possibilities
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    board = EngineBoard.from_fen(fen)
    
    print("--- Benchmark: Heuristic Evaluator ---")
    engine_heuristic = ChessEngine(max_depth=64, time_limit=5.0, use_nn=False)
    best_move_h, stats_h = engine_heuristic.get_best_move(board)
    print(f"Best Move: {best_move_h}")
    print(f"Depth Reached: {stats_h['depth']}")
    print(f"Nodes Evaluated: {stats_h['nodes']}")
    print(f"Time (s): {stats_h['time_s']}")
    print(f"Nodes Per Second (NPS): {stats_h['nps']}")
    
    print("\n--- Benchmark: NNUE Evaluator ---")
    engine_nnue = ChessEngine(max_depth=64, time_limit=5.0, use_nn=True)
    best_move_nn, stats_nn = engine_nnue.get_best_move(board)
    print(f"Best Move: {best_move_nn}")
    print(f"Depth Reached: {stats_nn['depth']}")
    print(f"Nodes Evaluated: {stats_nn['nodes']}")
    print(f"Time (s): {stats_nn['time_s']}")
    print(f"Nodes Per Second (NPS): {stats_nn['nps']}")

if __name__ == "__main__":
    run_benchmark()
