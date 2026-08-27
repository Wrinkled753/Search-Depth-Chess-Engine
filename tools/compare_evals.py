import sys
import random
from pathlib import Path

# Add the project root to the python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import chess
from engine.board import EngineBoard
from engine.engine import ChessEngine

# A set of diverse opening positions to prevent identical games
OPENING_POSITIONS = [
    # Standard starting position
    ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    # Italian Game (after 1.e4 e5 2.Nf3 Nc6 3.Bc4)
    ("Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    # Sicilian Defense (after 1.e4 c5)
    ("Sicilian Defense", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
    # Queen's Gambit (after 1.d4 d5 2.c4)
    ("Queen's Gambit", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2"),
    # King's Indian (after 1.d4 Nf6 2.c4 g6)
    ("King's Indian", "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
    # French Defense (after 1.e4 e6)
    ("French Defense", "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    # Caro-Kann (after 1.e4 c6)
    ("Caro-Kann", "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    # Ruy Lopez (after 1.e4 e5 2.Nf3 Nc6 3.Bb5)
    ("Ruy Lopez", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
]

MAX_MOVES_PER_GAME = 150
SEARCH_DEPTH = 3

def play_game(engine_white: ChessEngine, engine_black: ChessEngine, opening_name: str, fen: str) -> str:
    """
    Play a single game between two engines from a given position.
    Returns '1-0', '0-1', or '1/2-1/2'.
    """
    board = EngineBoard.from_fen(fen)
    move_count = 0
    
    while not board.is_game_over() and move_count < MAX_MOVES_PER_GAME:
        if board.turn() == chess.WHITE:
            best_move = engine_white.get_best_move(board)
        else:
            best_move = engine_black.get_best_move(board)
            
        if best_move is None:
            break
            
        board.push(best_move)
        move_count += 1
    
    if board.is_checkmate():
        # The side to move is checkmated
        return "0-1" if board.turn() == chess.WHITE else "1-0"
    else:
        return "1/2-1/2"

def run_comparison(num_games: int = 10):
    """
    Run a comparison match between heuristic and NN engines.
    Each game uses a random opening, and colors alternate.
    """
    print("=" * 60)
    print("  ChessEngine-LLM: Heuristic vs Neural Network Comparison")
    print("=" * 60)
    print(f"Games: {num_games} | Depth: {SEARCH_DEPTH} | Max moves/game: {MAX_MOVES_PER_GAME}")
    print()
    
    heuristic_engine = ChessEngine(depth=SEARCH_DEPTH, use_nn=False)
    nn_engine = ChessEngine(depth=SEARCH_DEPTH, use_nn=True)
    
    # Track results from the NN engine's perspective
    nn_wins = 0
    nn_losses = 0
    draws = 0
    
    results_log = []
    
    for i in range(num_games):
        # Pick a random opening
        opening_name, fen = random.choice(OPENING_POSITIONS)
        
        # Alternate colors: even games NN=White, odd games NN=Black
        nn_is_white = (i % 2 == 0)
        
        if nn_is_white:
            engine_w, engine_b = nn_engine, heuristic_engine
            color_str = "NN=White, Heuristic=Black"
        else:
            engine_w, engine_b = heuristic_engine, nn_engine
            color_str = "Heuristic=White, NN=Black"
        
        print(f"Game {i+1}/{num_games}: {opening_name} | {color_str}")
        result = play_game(engine_w, engine_b, opening_name, fen)
        
        # Interpret result from NN's perspective
        if result == "1-0":
            if nn_is_white:
                nn_wins += 1
                nn_result = "WIN"
            else:
                nn_losses += 1
                nn_result = "LOSS"
        elif result == "0-1":
            if nn_is_white:
                nn_losses += 1
                nn_result = "LOSS"
            else:
                nn_wins += 1
                nn_result = "WIN"
        else:
            draws += 1
            nn_result = "DRAW"
            
        results_log.append((i+1, opening_name, color_str, result, nn_result))
        print(f"  Result: {result} (NN: {nn_result})")
    
    # Summary table
    print()
    print("=" * 60)
    print("  RESULTS SUMMARY (from NN Engine perspective)")
    print("=" * 60)
    print(f"{'Game':<6} {'Opening':<20} {'Colors':<30} {'Result':<10} {'NN':<6}")
    print("-" * 72)
    for game_num, opening, colors, result, nn_res in results_log:
        print(f"{game_num:<6} {opening:<20} {colors:<30} {result:<10} {nn_res:<6}")
    print("-" * 72)
    print(f"NN Wins: {nn_wins} | Draws: {draws} | NN Losses: {nn_losses}")
    print(f"NN Score: {nn_wins + draws * 0.5} / {num_games}")
    print("=" * 60)

if __name__ == "__main__":
    run_comparison(num_games=10)
