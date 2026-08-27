import torch
import numpy as np
import chess
from engine.board import EngineBoard
from training.model import ChessEvalNet
from data.prepare_dataset import encode_board

class NNUEEvaluator:
    """
    Neural network-based position evaluator.
    Wraps the trained ChessEvalNet model and provides a callable interface
    identical to the hand-written evaluate() function.
    
    Output is in [-1, 1] range (tanh output). Mate/stalemate detection
    is handled upstream in search.py, so this evaluator only needs to
    score non-terminal positions.
    """
    
    def __init__(self, model_path: str = "models/eval_net.pt", hidden_size: int = 256, num_hidden_layers: int = 3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessEvalNet(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
    
    def __call__(self, board: EngineBoard) -> float:
        """
        Evaluate a board position using the neural network.
        Returns a float in [-1, 1]. Positive = white advantage.
        """
        features = encode_board(board._board)
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            score = self.model(tensor).item()
        
        return score
