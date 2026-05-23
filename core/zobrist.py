"""
    Program: zobrist.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: Zobrist hashing for transposition tables. A Zobrist hash is a
             64-bit integer uniquely representing a board position, updated
             incrementally via XOR on each make/unmake move.
"""

import random
from core.constants import WHITE_SHORT_RIGHT, WHITE_LONG_RIGHT, BLACK_SHORT_RIGHT, BLACK_LONG_RIGHT
from core.constants import (
    FLAG_QUIET, FLAG_DOUBLE_PUSH, FLAG_CAPTURE, FLAG_EP_CAPTURE,
    FLAG_SHORT_CASTLE, FLAG_LONG_CASTLE,
    FLAG_PROMOTE_KNIGHT, FLAG_PROMOTE_BISHOP,
    FLAG_PROMOTE_ROOK,   FLAG_PROMOTE_QUEEN,
    FLAG_PROMOTE_KNIGHT_CAPTURE, FLAG_PROMOTE_BISHOP_CAPTURE,
    FLAG_PROMOTE_ROOK_CAPTURE,   FLAG_PROMOTE_QUEEN_CAPTURE,
    get_from_square, get_to_square, get_flag,
    A1, C1, D1, E1, F1, G1, H1,
    A8, C8, D8, E8, F8, G8, H8,
    WHITE_SHORT_RIGHT, WHITE_LONG_RIGHT,
    BLACK_SHORT_RIGHT, BLACK_LONG_RIGHT,
)

# Seed for reproducibility
ZOBRIST_SEED = 505

random.seed(ZOBRIST_SEED)

def _rand64() -> int:
    return random.getrandbits(64)

PIECE_INDICES = {
    'white_pawns':   0,
    'white_knights': 1,
    'white_bishops': 2,
    'white_rooks':   3,
    'white_queens':  4,
    'white_king':    5,
    'black_pawns':   6,
    'black_knights': 7,
    'black_bishops': 8,
    'black_rooks':   9,
    'black_queens':  10,
    'black_king':    11,
}

ZOBRIST_PIECES = [[_rand64() for _ in range(64)] for _ in range(12)]

ZOBRIST_SIDE = _rand64() # XOR'd when black's turn

ZOBRIST_CASTLING = {
    WHITE_SHORT_RIGHT: _rand64(),
    WHITE_LONG_RIGHT:  _rand64(),
    BLACK_SHORT_RIGHT: _rand64(),
    BLACK_LONG_RIGHT:  _rand64(),
}

ZOBRIST_EN_PASSANT = [_rand64() for _ in range(8)]  # one per file (0-7)

def compute_hash(board) -> int:
    """
    Builds a Zobrist hash from scratch for a given position.
    Only called once at game start or when loading a position.
    All subsequent updates are handled incrementally by update_hash.
    """
    h = 0

    # Piece placement
    for piece_name, piece_index in PIECE_INDICES.items():
        bb = getattr(board, piece_name)
        temp = bb
        while temp:
            square = (temp & -temp).bit_length() - 1
            h ^= ZOBRIST_PIECES[piece_index][square]
            temp &= temp - 1

    if not board.white_to_move:
        h ^= ZOBRIST_SIDE

    # Castling rights
    for right, zobrist_value in ZOBRIST_CASTLING.items():
        if board.castling_rights & right:
            h ^= zobrist_value

    # En passant file
    if board.en_passant_square is not None:
        ep_file = board.en_passant_square % 8
        h ^= ZOBRIST_EN_PASSANT[ep_file]

    return h

def update_hash(h: int, move: int, board) -> int:
    """
    Incrementally updates a Zobrist hash for a given move.
    Called in make_move and unmake_move.
    Since XOR is its own inverse, the same function works for both.
    """

    from_sq = get_from_square(move)
    to_sq   = get_to_square(move)
    flag    = get_flag(move)

    color, piece = board.get_piece_at(from_sq)
    piece_index  = PIECE_INDICES[f"{color}_{piece}"]

    # Flip side to move
    h ^= ZOBRIST_SIDE

    # Remove en passant file if one was active before this move
    if board.en_passant_square is not None:
        h ^= ZOBRIST_EN_PASSANT[board.en_passant_square % 8]

    # Handle each flag
    if flag == FLAG_QUIET:
        h ^= ZOBRIST_PIECES[piece_index][from_sq]  # remove from source
        h ^= ZOBRIST_PIECES[piece_index][to_sq]    # place at destination

    elif flag == FLAG_DOUBLE_PUSH:
        h ^= ZOBRIST_PIECES[piece_index][from_sq]
        h ^= ZOBRIST_PIECES[piece_index][to_sq]
        # XOR in the new en passant file
        ep_file = (to_sq - 8 if color == 'white' else to_sq + 8) % 8
        h ^= ZOBRIST_EN_PASSANT[ep_file]

    elif flag == FLAG_CAPTURE:
        captured = board.get_piece_at(to_sq)
        if captured:
            cap_color, cap_piece = captured
            h ^= ZOBRIST_PIECES[PIECE_INDICES[f"{cap_color}_{cap_piece}"]][to_sq]
        h ^= ZOBRIST_PIECES[piece_index][from_sq]
        h ^= ZOBRIST_PIECES[piece_index][to_sq]

    elif flag == FLAG_EP_CAPTURE:
        h ^= ZOBRIST_PIECES[piece_index][from_sq]
        h ^= ZOBRIST_PIECES[piece_index][to_sq]
        # Remove the captured pawn
        captured_pawn_sq = to_sq - 8 if color == 'white' else to_sq + 8
        cap_index = PIECE_INDICES['black_pawns' if color == 'white' else 'white_pawns']
        h ^= ZOBRIST_PIECES[cap_index][captured_pawn_sq]

    elif flag == FLAG_SHORT_CASTLE:
        if color == 'white':
            h ^= ZOBRIST_PIECES[PIECE_INDICES['white_king']][E1]
            h ^= ZOBRIST_PIECES[PIECE_INDICES['white_king']][G1]
            h ^= ZOBRIST_PIECES[PIECE_INDICES['white_rooks']][H1]
            h ^= ZOBRIST_PIECES[PIECE_INDICES['white_rooks']][F1]
            h ^= ZOBRIST_CASTLING[WHITE_SHORT_RIGHT]
        else:
            h ^= ZOBRIST_PIECES[PIECE_INDICES['black_king']][E8]
            h ^= ZOBRIST_PIECES[PIECE_INDICES['black_king']][G8]
            h ^= ZOBRIST_PIECES[PIECE_INDICES['black_rooks']][H8]
            h ^= ZOBRIST_PIECES[PIECE_INDICES['black_rooks']][F8]
            h ^= ZOBRIST_CASTLING[BLACK_SHORT_RIGHT]

    elif flag == FLAG_LONG_CASTLE:
        if color == 'white':
            h ^= ZOBRIST_PIECES[PIECE_INDICES['white_king']][E1]
            h ^= ZOBRIST_PIECES[PIECE_INDICES['white_king']][C1]
            h ^= ZOBRIST_PIECES[PIECE_INDICES['white_rooks']][A1]
            h ^= ZOBRIST_PIECES[PIECE_INDICES['white_rooks']][D1]
            h ^= ZOBRIST_CASTLING[WHITE_LONG_RIGHT]
        else:
            h ^= ZOBRIST_PIECES[PIECE_INDICES['black_king']][E8]
            h ^= ZOBRIST_PIECES[PIECE_INDICES['black_king']][C8]
            h ^= ZOBRIST_PIECES[PIECE_INDICES['black_rooks']][A8]
            h ^= ZOBRIST_PIECES[PIECE_INDICES['black_rooks']][D8]
            h ^= ZOBRIST_CASTLING[BLACK_LONG_RIGHT]

    elif flag in (FLAG_PROMOTE_QUEEN,  FLAG_PROMOTE_ROOK,
                  FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_KNIGHT,
                  FLAG_PROMOTE_QUEEN_CAPTURE,  FLAG_PROMOTE_ROOK_CAPTURE,
                  FLAG_PROMOTE_BISHOP_CAPTURE, FLAG_PROMOTE_KNIGHT_CAPTURE):

        # Remove captured piece if capture-promotion
        if flag >= FLAG_PROMOTE_KNIGHT_CAPTURE:
            captured = board.get_piece_at(to_sq)
            if captured:
                cap_color, cap_piece = captured
                h ^= ZOBRIST_PIECES[PIECE_INDICES[f"{cap_color}_{cap_piece}"]][to_sq]

        # Remove the pawn
        h ^= ZOBRIST_PIECES[piece_index][from_sq]

        # Place the promoted piece
        promote_map = {
            FLAG_PROMOTE_QUEEN:          f"{color}_queens",
            FLAG_PROMOTE_ROOK:           f"{color}_rooks",
            FLAG_PROMOTE_BISHOP:         f"{color}_bishops",
            FLAG_PROMOTE_KNIGHT:         f"{color}_knights",
            FLAG_PROMOTE_QUEEN_CAPTURE:  f"{color}_queens",
            FLAG_PROMOTE_ROOK_CAPTURE:   f"{color}_rooks",
            FLAG_PROMOTE_BISHOP_CAPTURE: f"{color}_bishops",
            FLAG_PROMOTE_KNIGHT_CAPTURE: f"{color}_knights",
        }
        promoted_index = PIECE_INDICES[promote_map[flag]]
        h ^= ZOBRIST_PIECES[promoted_index][to_sq]

    # Update castling rights
    castling_squares = {
        E1: WHITE_SHORT_RIGHT | WHITE_LONG_RIGHT,
        H1: WHITE_SHORT_RIGHT,
        A1: WHITE_LONG_RIGHT,
        E8: BLACK_SHORT_RIGHT | BLACK_LONG_RIGHT,
        H8: BLACK_SHORT_RIGHT,
        A8: BLACK_LONG_RIGHT,
    }
    for sq, rights_affected in castling_squares.items():
        if (from_sq == sq or to_sq == sq):
            for right in (WHITE_SHORT_RIGHT, WHITE_LONG_RIGHT,
                          BLACK_SHORT_RIGHT, BLACK_LONG_RIGHT):
                if (rights_affected & right) and (board.castling_rights & right):
                    h ^= ZOBRIST_CASTLING[right]

    return h