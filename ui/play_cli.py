import sys
from pathlib import Path

# Add the project root to the python path so we can import engine
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import chess
from engine.board import EngineBoard
from engine.engine import ChessEngine

def play():
    print("Welcome to ChessEngine-LLM CLI!")
    print("You are playing White. Enter moves in UCI format (e.g., e2e4, g1f3).")
    
    board = EngineBoard()
    engine = ChessEngine(depth=4)
    
    while not board.is_game_over():
        print("\n" + str(board._board))
        print(f"\nEvaluation (approx): {engine.depth} depth search...")
        
        if board.turn() == chess.WHITE:
            # User turn
            while True:
                move_str = input("Your move: ").strip()
                if move_str.lower() in ['quit', 'exit', 'resign']:
                    print("You resigned. Game over!")
                    return
                try:
                    board.push_uci(move_str)
                    break
                except ValueError:
                    print("Invalid move. Try again. Make sure to use UCI format (e.g. e2e4).")
        else:
            # Engine turn
            print("Engine is thinking...")
            best_move = engine.get_best_move(board)
            if best_move:
                print(f"Engine plays: {best_move.uci()}")
                board.push(best_move)
            else:
                print("Engine cannot find a move! (Error/Mate)")
                break
                
    print("\nGame Over!")
    print("Result:", board._board.result())

if __name__ == "__main__":
    play()
