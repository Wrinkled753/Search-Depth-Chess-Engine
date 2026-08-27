import chess
import chess.pgn
import chess.polyglot
import io
from typing import Iterator, Optional

class EngineBoard:
    """
    A wrapper around python-chess Board to provide a stable API
    for the engine's search and evaluation layers.
    """

    def __init__(self, fen: Optional[str] = None):
        """
        Initialize the board with a FEN string or the standard starting position.
        """
        if fen:
            self._board = chess.Board(fen)
        else:
            self._board = chess.Board()

    @classmethod
    def from_fen(cls, fen: str) -> "EngineBoard":
        """
        Create an EngineBoard from a FEN string.
        """
        return cls(fen=fen)

    def to_fen(self) -> str:
        """
        Get the FEN representation of the current board state.
        """
        return self._board.fen()

    @classmethod
    def from_pgn(cls, pgn_string: str) -> "EngineBoard":
        """
        Create an EngineBoard from a PGN string.
        This sets the board to the final position in the PGN.
        """
        pgn_io = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            raise ValueError("Invalid PGN string")
        
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            
        engine_board = cls()
        engine_board._board = board
        return engine_board

    def to_pgn(self) -> str:
        """
        Return a PGN string representing the move history of this board.
        Note: This relies on the move stack present in the chess.Board.
        """
        game = chess.pgn.Game.from_board(self._board)
        exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
        return game.accept(exporter)

    def legal_moves(self) -> Iterator[chess.Move]:
        """
        Generate all legal moves in the current position.
        """
        return self._board.legal_moves

    def push(self, move: chess.Move) -> None:
        """
        Make a move on the board.
        """
        self._board.push(move)

    def push_uci(self, uci_move: str) -> None:
        """
        Make a move on the board using its UCI string representation (e.g., 'e2e4').
        """
        move = chess.Move.from_uci(uci_move)
        if move in self._board.legal_moves:
            self._board.push(move)
        else:
            raise ValueError(f"Illegal move: {uci_move}")

    def pop(self) -> chess.Move:
        """
        Undo the last move on the board.
        """
        return self._board.pop()

    def is_checkmate(self) -> bool:
        """
        Return True if the current side to move is in checkmate.
        """
        return self._board.is_checkmate()

    def is_check(self) -> bool:
        """
        Return True if the current side to move is in check.
        """
        return self._board.is_check()

    def is_stalemate(self) -> bool:
        """
        Return True if the current side to move is in stalemate.
        """
        return self._board.is_stalemate()
        
    def is_game_over(self) -> bool:
        """
        Return True if the game is over (checkmate, stalemate, draw, etc).
        """
        return self._board.is_game_over()

    def turn(self) -> chess.Color:
        """
        Return the color of the player whose turn it is.
        chess.WHITE (True) or chess.BLACK (False).
        """
        return self._board.turn

    def zobrist_hash(self) -> int:
        """
        Return the Zobrist hash of the current board state.
        Used for transposition table lookups.
        """
        return chess.polyglot.zobrist_hash(self._board)

