import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from engine.board import EngineBoard
from engine.engine import ChessEngine
from engine.search import negamax, SearchInfo, INF
from engine.evaluate import evaluate as heuristic_eval
from engine.transposition import TranspositionTable

def debug():
    fen = "r1bqkbnr/pppppppp/8/8/3Q4/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
    board = EngineBoard.from_fen(fen)
    
    print("Evaluating moves for black:")
    legal_moves = list(board.legal_moves())
    for move in legal_moves:
        board.push(move)
        info = SearchInfo(tt=TranspositionTable(), time_limit=10.0)
        # We are at depth 2 now for White
        score, _ = negamax(board, 2, -INF, INF, heuristic_eval, info)
        score = -score
        board.pop()
        
        if move.uci() in ['d8d4', 'd7d5', 'e7e5']:
            print(f"Move: {move.uci()} Score: {score}")

if __name__ == "__main__":
    debug()
