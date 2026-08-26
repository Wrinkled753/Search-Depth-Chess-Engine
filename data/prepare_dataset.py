import os
import json
import math
import torch
import chess
import numpy as np
from torch.utils.data import TensorDataset

INPUT_FILE = "data/raw_evals.jsonl"
OUTPUT_DIR = "data/processed"
CLIP_LIMIT = 1000.0
SCALE_K = 400.0

# 12 piece types mapped to index [0, 11]
PIECE_TO_IDX = {
    chess.WHITE: {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    },
    chess.BLACK: {
        chess.PAWN: 6, chess.KNIGHT: 7, chess.BISHOP: 8,
        chess.ROOK: 9, chess.QUEEN: 10, chess.KING: 11
    }
}

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encodes the board into a 769-bit vector.
    768 bits for piece-square combinations + 1 bit for turn.
    """
    features = np.zeros(769, dtype=np.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_idx = PIECE_TO_IDX[piece.color][piece.piece_type]
            # 64 squares * 12 pieces = 768 flat index
            feature_idx = square * 12 + piece_idx
            features[feature_idx] = 1.0
            
    # Turn bit: 1.0 for White, 0.0 for Black
    features[768] = 1.0 if board.turn == chess.WHITE else 0.0
    return features

def prepare_data(input_file, output_dir):
    print(f"Reading from {input_file}...")
    X_list = []
    Y_list = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            fen = data["fen"]
            score_type = data["type"]
            score_val = data["value"]
            
            # Map mate to +/- clip limit, and clamp cp to +/- clip limit
            if score_type == "mate":
                score = CLIP_LIMIT if score_val > 0 else -CLIP_LIMIT
                # If mate is 0, it means the game is already over (checkmate on board)
                if score_val == 0:
                    board = chess.Board(fen)
                    score = -CLIP_LIMIT if board.turn == chess.WHITE else CLIP_LIMIT
            else: # "cp"
                score = max(-CLIP_LIMIT, min(CLIP_LIMIT, float(score_val)))
                
            # Normalize to [-1, 1] using tanh(score / K)
            normalized_score = math.tanh(score / SCALE_K)
            
            board = chess.Board(fen)
            features = encode_board(board)
            
            X_list.append(features)
            Y_list.append([normalized_score])
            
            if len(X_list) % 50000 == 0:
                print(f"Processed {len(X_list)} samples...")

    print("Converting to PyTorch tensors...")
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float32)
    
    # Split into 80/10/10
    total = len(X_list)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    print("Splitting datasets...")
    X_train, Y_train = X_tensor[:train_end], Y_tensor[:train_end]
    X_val, Y_val = X_tensor[train_end:val_end], Y_tensor[train_end:val_end]
    X_test, Y_test = X_tensor[val_end:], Y_tensor[val_end:]
    
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    test_ds = TensorDataset(X_test, Y_test)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {output_dir}...")
    torch.save(train_ds, os.path.join(output_dir, "train.pt"))
    torch.save(val_ds, os.path.join(output_dir, "val.pt"))
    torch.save(test_ds, os.path.join(output_dir, "test.pt"))
    
    print(f"Done! Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

if __name__ == "__main__":
    prepare_data(INPUT_FILE, OUTPUT_DIR)
