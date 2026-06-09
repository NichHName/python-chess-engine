"""
    Program: base_layers.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: Evaluation function for the chess engine. Returns a score
             from white's perspective — positive favors white, negative
             favors black, zero is equal. The search calls evaluate()
             exclusively; internal functions are implementation details.
"""

from core.bitboard import Board

# ====================================================
# Piece values (centipawns)
# ====================================================
PAWN_VALUE   = 100
KNIGHT_VALUE = 320
BISHOP_VALUE = 330
ROOK_VALUE   = 500
QUEEN_VALUE  = 900

# ====================================================
# Piece-square tables
# Each table is from white's perspective, rank 1 to rank 8
# (index 0 = A1, index 63 = H8)
# These provide positional bonuses/penalties on top of material
# ====================================================
PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_MIDDLE_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

# ====================================================
# Helpers
# ====================================================
def _popcount(bb: int) -> int:
    return bin(bb).count('1')

def _evaluate_piece_square(bb: int, table: list, is_black: bool) -> int:
    """
    Sums the piece-square table values for all pieces on the bitboard.
    Black's table is mirrored vertically (rank 8 becomes rank 1).
    """
    score = 0
    temp = bb
    while temp:
        square = (temp & -temp).bit_length() - 1
        if is_black:
            # Mirror vertically: rank 8 -> rank 1
            rank = square // 8
            file = square % 8
            mirrored = (7 - rank) * 8 + file
            score += table[mirrored]
        else:
            score += table[square]
        temp &= temp - 1
    return score

# ====================================================
# Material counting
# ====================================================
def count_material(board: Board) -> int:
    """
    Returns raw material balance from white's perspective.
    Kept separate so the search can use it independently
    (e.g. insufficient material detection).
    """
    white_material = (
        _popcount(board.white_pawns)   * PAWN_VALUE   +
        _popcount(board.white_knights) * KNIGHT_VALUE +
        _popcount(board.white_bishops) * BISHOP_VALUE +
        _popcount(board.white_rooks)   * ROOK_VALUE   +
        _popcount(board.white_queens)  * QUEEN_VALUE
    )

    black_material = (
        _popcount(board.black_pawns)   * PAWN_VALUE   +
        _popcount(board.black_knights) * KNIGHT_VALUE +
        _popcount(board.black_bishops) * BISHOP_VALUE +
        _popcount(board.black_rooks)   * ROOK_VALUE   +
        _popcount(board.black_queens)  * QUEEN_VALUE
    )

    return white_material - black_material

# ====================================================
# Positional evaluation
# ====================================================
def _evaluate_position(board: Board) -> int:
    """
    Returns positional score from white's perspective using
    piece-square tables.
    """
    score = 0

    # White positional bonuses
    score += _evaluate_piece_square(board.white_pawns,   PAWN_TABLE,   is_black=False)
    score += _evaluate_piece_square(board.white_knights, KNIGHT_TABLE, is_black=False)
    score += _evaluate_piece_square(board.white_bishops, BISHOP_TABLE, is_black=False)
    score += _evaluate_piece_square(board.white_rooks,   ROOK_TABLE,   is_black=False)
    score += _evaluate_piece_square(board.white_queens,  QUEEN_TABLE,  is_black=False)
    score += _evaluate_piece_square(board.white_king,    KING_MIDDLE_TABLE, is_black=False)

    # Black positional bonuses (subtracted from score)
    score -= _evaluate_piece_square(board.black_pawns,   PAWN_TABLE,   is_black=True)
    score -= _evaluate_piece_square(board.black_knights, KNIGHT_TABLE, is_black=True)
    score -= _evaluate_piece_square(board.black_bishops, BISHOP_TABLE, is_black=True)
    score -= _evaluate_piece_square(board.black_rooks,   ROOK_TABLE,   is_black=True)
    score -= _evaluate_piece_square(board.black_queens,  QUEEN_TABLE,  is_black=True)
    score -= _evaluate_piece_square(board.black_king,    KING_MIDDLE_TABLE, is_black=True)

    return score

# ====================================================
# Top-level evaluation function
# ====================================================
def evaluate(board: Board) -> int:
    """
    Returns a score from white's perspective.
    Positive = white is better, negative = black is better.
    """
    return count_material(board) + _evaluate_position(board)