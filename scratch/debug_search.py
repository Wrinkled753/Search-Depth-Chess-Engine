import chess
from engine.board import EngineBoard
from engine.engine import ChessEngine

fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
board = EngineBoard.from_fen(fen)
engine = ChessEngine(max_depth=6)
moves = list(board.legal_moves())
def move_score(move: chess.Move) -> int:
    score = 0
    if board._board.is_capture(move):
        score += 10
    if move.promotion:
        score += 5
    return score

moves.sort(key=move_score, reverse=True)
for m in moves:
    board.push(m)
    from engine.search import minimax
    score, _ = minimax(board, depth=1, alpha=-float('inf'), beta=float('inf'), maximizing_player=False)
    print(f"Move: {m}, Score: {score}")
    board.pop()

print("Best move:", engine.get_best_move(board))
