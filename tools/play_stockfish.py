"""
Tournament script to benchmark ChessEngine-LLM against Stockfish.
Plays a series of games to estimate the Elo rating of both the 
Heuristic and Neural Network evaluation backends.
"""
import sys
import shutil
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import chess
import chess.engine
from engine.board import EngineBoard
from engine.engine import ChessEngine

# Configuration
STOCKFISH_ELO = 1320  # Minimum supported by Stockfish is usually 1320
NUM_GAMES_PER_BACKEND = 10
TIME_LIMIT = 0.1  # 100ms per move for fast games

def play_match(our_engine: ChessEngine, sf_engine: chess.engine.SimpleEngine, 
               our_color: chess.Color) -> float:
    """
    Plays a single game against Stockfish. 
    Returns 1.0 for win, 0.5 for draw, 0.0 for loss.
    """
    board = chess.Board()
    engine_board = EngineBoard()
    
    while not board.is_game_over():
        if board.turn == our_color:
            # Our engine's turn
            engine_board._board = board.copy()
            # Suppress standard output for cleaner logs
            best_move, _ = our_engine.get_best_move(engine_board, time_limit=TIME_LIMIT)
            if best_move is None:
                print("Error: Engine returned no move.")
                return 0.0
            board.push(best_move)
        else:
            # Stockfish's turn
            result = sf_engine.play(board, chess.engine.Limit(time=TIME_LIMIT))
            if result.move is None:
                print("Error: Stockfish returned no move.")
                return 1.0
            board.push(result.move)
            
    # Game over
    outcome = board.outcome()
    if outcome.winner == our_color:
        return 1.0
    elif outcome.winner is None:
        return 0.5
    else:
        return 0.0

def run_tournament(use_nn: bool, sf_path: str):
    """Runs a mini-tournament for a specific backend against Stockfish."""
    backend_name = "Neural Network" if use_nn else "Heuristic"
    print(f"\n{'='*50}")
    print(f" Starting Tournament: {backend_name} Engine vs Stockfish")
    print(f" Stockfish Elo: {STOCKFISH_ELO} | Time Limit: {TIME_LIMIT}s/move")
    print(f"{'='*50}")
    
    our_engine = ChessEngine(max_depth=64, time_limit=TIME_LIMIT, use_nn=use_nn)
    
    # Initialize Stockfish with restricted Elo
    sf_engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    sf_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": STOCKFISH_ELO})
    
    wins = draws = losses = 0
    
    for i in range(NUM_GAMES_PER_BACKEND):
        # Alternate colors
        our_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        color_str = "White" if our_color == chess.WHITE else "Black"
        
        print(f"Game {i+1}/{NUM_GAMES_PER_BACKEND} (Playing as {color_str})... ", end="", flush=True)
        
        score = play_match(our_engine, sf_engine, our_color)
        
        if score == 1.0:
            print("WIN")
            wins += 1
        elif score == 0.5:
            print("DRAW")
            draws += 1
        else:
            print("LOSS")
            losses += 1
            
    sf_engine.quit()
    
    # Calculate stats
    total_score = wins + (draws * 0.5)
    win_rate = total_score / NUM_GAMES_PER_BACKEND
    
    print(f"\n--- {backend_name} Results ---")
    print(f"Wins: {wins} | Draws: {draws} | Losses: {losses}")
    print(f"Win Rate: {win_rate*100:.1f}% against Elo {STOCKFISH_ELO}")
    
    # Very rough Elo estimate formula based on winrate
    # E_diff = -400 * log10(1/score - 1)
    if win_rate > 0 and win_rate < 1:
        import math
        elo_diff = -400 * math.log10(1/win_rate - 1)
        estimated_elo = STOCKFISH_ELO + elo_diff
        print(f"Estimated Elo: {estimated_elo:.0f}")
    elif win_rate == 1:
        print(f"Estimated Elo: > {STOCKFISH_ELO + 400}")
    else:
        print(f"Estimated Elo: < {STOCKFISH_ELO - 400}")

if __name__ == "__main__":
    sf_path = shutil.which("stockfish")
    if not sf_path:
        print("ERROR: 'stockfish' executable not found in PATH.")
        print("Please install Stockfish to run this benchmark.")
        print("Ubuntu: sudo apt install stockfish")
        print("Mac: brew install stockfish")
        print("Windows: Download from stockfishchess.org and add to PATH")
        sys.exit(1)
        
    print(f"Found Stockfish at: {sf_path}")
    
    # Test Heuristic Engine
    run_tournament(use_nn=False, sf_path=sf_path)
    
    # Test Neural Network Engine
    run_tournament(use_nn=True, sf_path=sf_path)
