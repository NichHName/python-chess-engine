"""
    Program: move_ordering.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: Move ordering for the search. Scores moves so that
             the search sees the most promising moves first, maximizing
             alpha-beta pruning efficiency.
             
             Ordering priority:
             1. Captures (scored by MVV-LVA)
             2. Quiet moves (scored 0)
"""

from core.bitboard import Board
from core.constants import get_from_square, get_to_square, get_flag
from core.constants import (
    FLAG_CAPTURE, FLAG_EP_CAPTURE,
    FLAG_PROMOTE_QUEEN, FLAG_PROMOTE_ROOK,
    FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_KNIGHT,
    FLAG_PROMOTE_QUEEN_CAPTURE, FLAG_PROMOTE_ROOK_CAPTURE,
    FLAG_PROMOTE_BISHOP_CAPTURE, FLAG_PROMOTE_KNIGHT_CAPTURE,
)
from evaluator.base_layers import (
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE,
    ROOK_VALUE, QUEEN_VALUE
)

# =========================================================
# Piece value lookup for MVV-LVA
# Maps piece attribute name to centipawn value
# =========================================================
PIECE_VALUES = {
    'pawns':   PAWN_VALUE,
    'knights': KNIGHT_VALUE,
    'bishops': BISHOP_VALUE,
    'rooks':   ROOK_VALUE,
    'queens':  QUEEN_VALUE,
    'king':    20000,  # king capture shouldn't occur but handled for safety
}

# MVV-LVA score offset — ensures all captures score above quiet moves
CAPTURE_OFFSET = 10000

# Promotion bonus — promotions score above quiet moves
PROMOTION_BONUS = 9000

# =========================================================
# MVV-LVA scoring
# =========================================================
def _mvv_lva_score(board: Board, move: int) -> int:
    """
    Most Valuable Victim - Least Valuable Attacker.
    Returns a score for a capture move.
    Higher = search this capture first.
    """
    from_sq = get_from_square(move)
    to_sq   = get_to_square(move)

    # Get attacker value
    attacker = board.get_piece_at(from_sq)
    if attacker is None:
        return 0
    _, attacker_piece = attacker
    attacker_value = PIECE_VALUES.get(attacker_piece, 0)

    # Get victim value
    victim = board.get_piece_at(to_sq)
    if victim is None:
        return 0
    _, victim_piece = victim
    victim_value = PIECE_VALUES.get(victim_piece, 0)

    # High victim value, low attacker value = high score
    return CAPTURE_OFFSET + victim_value - attacker_value

# =========================================================
# Move score
# =========================================================
def score_move(board: Board, move: int) -> int:
    """
    Returns a score for a single move.
    Higher score = search first.
    """
    flag = get_flag(move)

    # Capture-promotions
    if flag in (FLAG_PROMOTE_QUEEN_CAPTURE, FLAG_PROMOTE_ROOK_CAPTURE,
                FLAG_PROMOTE_BISHOP_CAPTURE, FLAG_PROMOTE_KNIGHT_CAPTURE):
        return CAPTURE_OFFSET + PROMOTION_BONUS

    # Quiet promotions
    if flag == FLAG_PROMOTE_QUEEN:
        return PROMOTION_BONUS
    if flag in (FLAG_PROMOTE_ROOK, FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_KNIGHT):
        return PROMOTION_BONUS - 100

    # En passant — treat as capture of pawn
    if flag == FLAG_EP_CAPTURE:
        from_sq = get_from_square(move)
        attacker = board.get_piece_at(from_sq)
        if attacker:
            _, attacker_piece = attacker
            attacker_value = PIECE_VALUES.get(attacker_piece, 0)
            return CAPTURE_OFFSET + PAWN_VALUE - attacker_value

    # Regular captures
    if flag == FLAG_CAPTURE:
        return _mvv_lva_score(board, move)

    # Quiet moves
    return 0

# =========================================================
# Order moves
# =========================================================
def order_moves(board: Board, moves: list) -> list:
    """
    Returns moves sorted by score, highest first.
    Called once per node in the search.
    """
    return sorted(moves, key=lambda move: score_move(board, move), reverse=True)