"""
    Program: minimax.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: Negamax search with alpha-beta pruning and quiescence search.
             Returns the best move for the current side to move.
"""

from core.bitboard import Board
from core.move_gen import get_all_legal_moves, is_square_attacked
from core.constants import get_flag, FLAG_CAPTURE, FLAG_EP_CAPTURE
from core.constants import (
    FLAG_PROMOTE_QUEEN_CAPTURE, FLAG_PROMOTE_ROOK_CAPTURE,
    FLAG_PROMOTE_BISHOP_CAPTURE, FLAG_PROMOTE_KNIGHT_CAPTURE,
    FLAG_PROMOTE_QUEEN, FLAG_PROMOTE_ROOK,
    FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_KNIGHT,
)
from evaluator.base_layers import evaluate
from evaluator.move_ordering import order_moves
from evaluator.train import load_network_for_engine, network_evaluate
from evaluator.eval_router import evaluate

_network = load_network_for_engine('models/network')

def get_evaluation(board):
    net_score = network_evaluate(_network, board)
    return net_score if net_score is not None else evaluate(board)

# =========================================================
# Constants
# =========================================================
CHECKMATE_SCORE  = 100000
STALEMATE_SCORE  = 0
MAX_QUIESCENCE_DEPTH = 8

# =========================================================
# Capture filter
# =========================================================
CAPTURE_FLAGS = (
    FLAG_CAPTURE,
    FLAG_EP_CAPTURE,
    FLAG_PROMOTE_QUEEN_CAPTURE,
    FLAG_PROMOTE_ROOK_CAPTURE,
    FLAG_PROMOTE_BISHOP_CAPTURE,
    FLAG_PROMOTE_KNIGHT_CAPTURE,
)

PROMOTION_FLAGS = (
    FLAG_PROMOTE_QUEEN,
    FLAG_PROMOTE_ROOK,
    FLAG_PROMOTE_BISHOP,
    FLAG_PROMOTE_KNIGHT,
)

def _is_capture(move: int) -> bool:
    return get_flag(move) in CAPTURE_FLAGS

def _is_promotion(move: int) -> bool:
    return get_flag(move) in PROMOTION_FLAGS

# =========================================================
# Quiescence search
# =========================================================
def quiescence(board: Board, alpha: int, beta: int, depth: int = MAX_QUIESCENCE_DEPTH) -> int:
    """
    Searches captures and promotions until the position is quiet.
    Prevents the horizon effect by not evaluating mid-capture sequences.
    Uses stand-pat pruning — if the current position already beats beta,
    we don't need to search further.
    """
    # Stand pat — evaluate current position from side to move's perspective
    stand_pat = evaluate(board)
    if not board.white_to_move:
        stand_pat = -stand_pat

    # Stand pat pruning
    if stand_pat >= beta:
        return beta

    if stand_pat > alpha:
        alpha = stand_pat

    # Depth limit on quiescence search
    if depth == 0:
        return alpha

    # Generate all legal moves and filter to captures and promotions
    moves = get_all_legal_moves(board)
    noisy_moves = [m for m in moves if _is_capture(m) or _is_promotion(m)]

    if not noisy_moves:
        return alpha

    # Order captures by MVV-LVA
    noisy_moves = order_moves(board, noisy_moves)

    for move in noisy_moves:
        board.make_move(move)
        score = -quiescence(board, -beta, -alpha, depth - 1)
        board.unmake_move(move)

        if score >= beta:
            return beta

        if score > alpha:
            alpha = score

    return alpha

# =========================================================
# Negamax with alpha-beta pruning
# =========================================================
def negamax(board: Board, depth: int, alpha: int, beta: int) -> int:
    """
    Returns the best score achievable from the current position,
    from the perspective of the side to move.
    """
    # Repetition detection
    if board.repetition_table.get(board.zobrist_hash, 0) >= 2:
        return STALEMATE_SCORE

    # Drop into quiescence at depth 0
    if depth == 0:
        return quiescence(board, alpha, beta)

    moves = get_all_legal_moves(board)

    # Terminal node
    if not moves:
        king    = board.white_king if board.white_to_move else board.black_king
        king_sq = (king & -king).bit_length() - 1
        in_check = is_square_attacked(board, king_sq, attacked_by_white=not board.white_to_move)

        if in_check:
            return -(CHECKMATE_SCORE - depth)
        else:
            return STALEMATE_SCORE

    # Order moves
    moves = order_moves(board, moves)

    best = -CHECKMATE_SCORE

    for move in moves:
        board.make_move(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.unmake_move(move)

        if score > best:
            best = score

        if best > alpha:
            alpha = best

        if alpha >= beta:
            break

    return best

# =========================================================
# Root level search
# =========================================================
def get_best_move(board: Board, depth: int):
    """
    Returns the best move and score for the current side to move.
    """
    best_move  = None
    best_score = -CHECKMATE_SCORE
    alpha      = -CHECKMATE_SCORE
    beta       =  CHECKMATE_SCORE

    moves = get_all_legal_moves(board)
    moves = order_moves(board, moves)

    for move in moves:
        board.make_move(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.unmake_move(move)

        if score > best_score:
            best_score = score
            best_move  = move

        if score > alpha:
            alpha = score

    return best_move, best_score