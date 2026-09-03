import torch
import numpy as np
import chess
from engine.board import EngineBoard
from training.model import ChessEvalNet

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

class NNUEEvaluator:
    """
    Neural network-based position evaluator.
    Wraps the trained ChessEvalNet model but uses a pure NumPy forward pass 
    for extremely fast batch=1 inference (eliminates PyTorch dispatch overhead).
    """
    
    def __init__(self, model_path: str = "models/eval_net.pt", hidden_size: int = 256, num_hidden_layers: int = 3):
        # Load weights using PyTorch
        device = torch.device("cpu")
        model = ChessEvalNet(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        # Extract weights and biases to numpy arrays for manual forward pass
        # The model is an nn.Sequential: Linear, ReLU, Linear, ReLU, Linear, ReLU, Linear, Tanh
        layers = list(model.model.children())
        
        # Layer 1
        self.W1 = layers[0].weight.detach().numpy()
        self.b1 = layers[0].bias.detach().numpy()
        
        # Layer 2
        self.W2 = layers[2].weight.detach().numpy()
        self.b2 = layers[2].bias.detach().numpy()
        
        # Layer 3
        self.W3 = layers[4].weight.detach().numpy()
        self.b3 = layers[4].bias.detach().numpy()
        
        # Layer 4 (Output)
        self.W4 = layers[6].weight.detach().numpy()
        self.b4 = layers[6].bias.detach().numpy()
        
        # Shared memory buffer for encoding
        self._buf = np.zeros(769, dtype=np.float32)
    
    def __call__(self, board: EngineBoard) -> float:
        """
        Evaluate a board position using the neural network.
        Returns a float in [-1, 1]. Positive = white advantage.
        """
        buf = self._buf
        buf.fill(0.0)
        
        for square, piece in board._board.piece_map().items():
            buf[square * 12 + PIECE_TO_IDX[piece.color][piece.piece_type]] = 1.0
            
        buf[768] = 1.0 if board._board.turn == chess.WHITE else 0.0
        
        # Manual numpy forward pass (much faster for single inference)
        # Layer 1
        h = np.dot(self.W1, buf) + self.b1
        h = np.maximum(h, 0) # ReLU
        
        # Layer 2
        h = np.dot(self.W2, h) + self.b2
        h = np.maximum(h, 0) # ReLU
        
        # Layer 3
        h = np.dot(self.W3, h) + self.b3
        h = np.maximum(h, 0) # ReLU
        
        # Layer 4
        h = np.dot(self.W4, h) + self.b4
        
        # Output Activation
        return float(np.tanh(h[0]))


