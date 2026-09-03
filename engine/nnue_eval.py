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
    Wraps the trained ChessEvalNet model and provides a callable interface.
    Uses CPU-only inference with shared numpy/tensor memory (zero-copy).
    """
    
    def __init__(self, model_path: str = "models/eval_net.pt", hidden_size: int = 256, num_hidden_layers: int = 3):
        self.device = torch.device("cpu")
        model = ChessEvalNet(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        
        self.model = model
        
        # Shared memory buffer: numpy array and torch tensor point to same data
        self._buf = np.zeros(769, dtype=np.float32)
        self._tensor = torch.from_numpy(self._buf).unsqueeze(0)  # shape (1, 769), shares memory
    
    @torch.no_grad()
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
        
        # No copy needed — tensor shares memory with numpy buffer
        return self.model(self._tensor).item()


