"""
    Program: move_gen.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: The purpose of this program is to handle the core logic of generating legal moves 
             on a bit-represented chess board. We use bit logic to maximize efficiency. The end
             goal is the get_legal_moves() function, where we input a bitboard and get all legal
             moves.
"""

from typing import List
from core.bitboard import Board
from core.constants import RANK_2, RANK_7
from core.constants import A1, B1, C1, D1, E1, F1, G1, H1, A8, B8, C8, D8, E8, F8, G8, H8
from core.constants import get_from_square, get_to_square, get_flag, encode_move
from core.constants import FLAG_CAPTURE, FLAG_DOUBLE_PUSH, FLAG_EP_CAPTURE, FLAG_LONG_CASTLE, FLAG_PROMOTE_BISHOP,\
                           FLAG_PROMOTE_BISHOP_CAPTURE, FLAG_PROMOTE_KNIGHT, FLAG_PROMOTE_KNIGHT_CAPTURE, FLAG_PROMOTE_QUEEN,\
                           FLAG_PROMOTE_QUEEN_CAPTURE, FLAG_PROMOTE_ROOK, FLAG_PROMOTE_ROOK_CAPTURE, FLAG_QUIET, FLAG_SHORT_CASTLE,\
                           WHITE_SHORT_RIGHT, WHITE_LONG_RIGHT, BLACK_SHORT_RIGHT, BLACK_LONG_RIGHT,\
                           WHITE_SHORT_EMPTY_MASK, WHITE_LONG_EMPTY_MASK, BLACK_SHORT_EMPTY_MASK, BLACK_LONG_EMPTY_MASK
from core.constants import RANK_1,RANK_2,RANK_3,RANK_4,RANK_5,RANK_6,RANK_7,RANK_8,\
                           FILE_A,FILE_B,FILE_D,FILE_E,FILE_F,FILE_G,FILE_H
from core.constants import MASK_64
from core.magic import get_bishop_attacks, get_rook_attacks

# =========================================================
# Precompute tables
# =========================================================
def _precompute_king_attacks() -> List[int]:

    attacks = [0] * 64
    for square in range(64):

        king = 1 << square

        up = (king << 8) & MASK_64
        down = (king >> 8)
        right = (king << 1) & ~FILE_A & MASK_64
        left = (king >> 1) & ~FILE_H
        ur = (king << 9) & MASK_64 & ~FILE_A
        ul = (king << 7) & MASK_64 & ~FILE_H
        dr = (king >> 7) & ~FILE_A
        dl = (king >> 9) & ~FILE_H

        attacks[square] = up | down | right | left | ur | ul | dr | dl

    return attacks    

def _precompute_knight_attacks() -> List[int]:
    attacks = [0] * 64
    for square in range(64):
        knight = 1 << square
        
        # Calculate the math (THIS IS THE ONLY TIME WE DO MATH)
        uul = (knight << 15) & ~FILE_H & MASK_64
        uur = (knight << 17) & ~FILE_A & MASK_64
        ull = (knight << 6)  & ~FILE_H & ~FILE_G & MASK_64
        urr = (knight << 10) & ~FILE_A & ~FILE_B & MASK_64
        ddl = (knight >> 17) & ~FILE_H
        ddr = (knight >> 15) & ~FILE_A
        dll = (knight >> 10) & ~FILE_H & ~FILE_G
        drr = (knight >> 6)  & ~FILE_A & ~FILE_B
        
        # Save all the jumps for this square into the array
        attacks[square] = uul | uur | ull | urr | ddl | ddr | dll | drr
        
    return attacks

def _precompute_white_pawn_attacks() -> List[int]:
    attacks = [0] * 64
    for square in range(64):
        pawn = 1 << square
        
        # White captures Up (+8). Left is +7, Right is +9
        up_left  = (pawn << 7) & ~FILE_H & MASK_64
        up_right = (pawn << 9) & ~FILE_A & MASK_64
        
        attacks[square] = up_left | up_right
        
    return attacks

def _precompute_black_pawn_attacks() -> List[int]:
    attacks = [0] * 64
    for square in range(64):
        pawn = 1 << square
        
        # Black captures Down (-8). Left is -9, Right is -7
        down_left  = (pawn >> 9) & ~FILE_H
        down_right = (pawn >> 7) & ~FILE_A
        
        attacks[square] = down_left | down_right
        
    return attacks

# Create the attack tables at the beginning of the game (save compute time during the game)
KING_ATTACK_TABLE       = _precompute_king_attacks()
KNIGHT_ATTACK_TABLE     = _precompute_knight_attacks()
WHITE_PAWN_ATTACK_TABLE = _precompute_white_pawn_attacks()
BLACK_PAWN_ATTACK_TABLE = _precompute_black_pawn_attacks()

# =========================================================
# Calculate moves on the board
# =========================================================
def get_white_pawn_moves(board: Board) -> List[int]:
    """
    Get all possible pawn moves for white.
    """
    moves = []

    empty_squares  = ~board.all_pieces & MASK_64
    single_pushes  = (board.white_pawns << 8) & empty_squares
    double_pushes  = (single_pushes << 8) & empty_squares & RANK_4
    left_captures  = (board.white_pawns << 7) & board.black_pieces & ~FILE_H
    right_captures = (board.white_pawns << 9) & board.black_pieces & ~FILE_A
    en_passants = None # TODO

    temp_singles = single_pushes
    while temp_singles:
        to_square = (temp_singles & -temp_singles).bit_length() - 1
        from_square = to_square - 8

        if (1 << to_square) & RANK_8:
            moves.append(encode_move(from_square, to_square, FLAG_PROMOTE_QUEEN))
            moves.append(encode_move(from_square, to_square, FLAG_PROMOTE_ROOK))
            moves.append(encode_move(from_square, to_square, FLAG_PROMOTE_BISHOP))
            moves.append(encode_move(from_square, to_square, FLAG_PROMOTE_KNIGHT))
        else:
            moves.append(encode_move(from_square, to_square, FLAG_QUIET))

        temp_singles &= temp_singles - 1
    
    temp_doubles = double_pushes
    while temp_doubles:
        to_square = (temp_doubles & -temp_doubles).bit_length() - 1
        from_square = to_square - 16
        moves.append(encode_move(from_square, to_square, FLAG_DOUBLE_PUSH))
        temp_doubles &= temp_doubles - 1

    temp_lefts = left_captures
    while temp_lefts:
        to_square = (temp_lefts & -temp_lefts).bit_length() - 1
        from_square = to_square - 7
        moves.append(encode_move(from_square, to_square, FLAG_CAPTURE))
        temp_lefts &= temp_lefts - 1

    temp_rights = right_captures
    while temp_rights:
        to_square = (temp_rights & -temp_rights).bit_length() - 1
        from_square = to_square - 9
        moves.append(encode_move(from_square, to_square, FLAG_CAPTURE))
        temp_rights &= temp_rights - 1
    
    if board.en_passant_square:
        ep_sq = board.en_passant_square
        # Which white pawns can attack the en passant square?
        ep_attackers = BLACK_PAWN_ATTACK_TABLE[ep_sq] & board.white_pawns
        temp_ep = ep_attackers
        while temp_ep:
            from_square = (temp_ep & -temp_ep).bit_length() - 1
            moves.append(encode_move(from_square, ep_sq, FLAG_EP_CAPTURE))
            temp_ep &= temp_ep - 1

    return moves

def get_black_pawn_moves(board: Board) -> List[int]:
    """
    Get all possible pawn moves for black.
    """
    moves = []

    empty_squares  = ~board.all_pieces & MASK_64
    single_pushes  = (board.black_pawns >> 8) & empty_squares
    double_pushes  = (single_pushes >> 8) & empty_squares & RANK_5
    left_captures  = (board.black_pawns >> 9) & board.white_pieces & ~FILE_H
    right_captures = (board.black_pawns >> 7) & board.white_pieces & ~FILE_A
    en_passants = None # TODO

    temp_singles = single_pushes
    while temp_singles:
        to_square = (temp_singles & -temp_singles).bit_length() - 1
        from_square = to_square + 8

        if (1 << to_square) & RANK_1:
            moves.append(encode_move(from_square, to_square, FLAG_PROMOTE_QUEEN))
            moves.append(encode_move(from_square, to_square, FLAG_PROMOTE_ROOK))
            moves.append(encode_move(from_square, to_square, FLAG_PROMOTE_BISHOP))
            moves.append(encode_move(from_square, to_square, FLAG_PROMOTE_KNIGHT))
        else:
            moves.append(encode_move(from_square, to_square, FLAG_QUIET))

        temp_singles &= temp_singles - 1
    
    temp_doubles = double_pushes
    while temp_doubles:
        to_square = (temp_doubles & -temp_doubles).bit_length() - 1
        from_square = to_square + 16
        moves.append(encode_move(from_square, to_square, FLAG_DOUBLE_PUSH))
        temp_doubles &= temp_doubles - 1

    temp_lefts = left_captures
    while temp_lefts:
        to_square = (temp_lefts & -temp_lefts).bit_length() - 1
        from_square = to_square + 9
        moves.append(encode_move(from_square, to_square, FLAG_CAPTURE))
        temp_lefts &= temp_lefts - 1

    temp_rights = right_captures
    while temp_rights:
        to_square = (temp_rights & -temp_rights).bit_length() - 1
        from_square = to_square + 7
        moves.append(encode_move(from_square, to_square, FLAG_CAPTURE))
        temp_rights &= temp_rights - 1
    
    if board.en_passant_square:
        ep_sq = board.en_passant_square
        # Which white pawns can attack the en passant square?
        ep_attackers = WHITE_PAWN_ATTACK_TABLE[ep_sq] & board.white_pawns
        temp_ep = ep_attackers
        while temp_ep:
            from_square = (temp_ep & -temp_ep).bit_length() - 1
            moves.append(encode_move(from_square, ep_sq, FLAG_EP_CAPTURE))
            temp_ep &= temp_ep - 1

    return moves

def get_knight_moves(knights_board: int, friendlies:int, all_pieces: int) -> List[int]:
    moves = []
    
    # Loop through the knights one by one using Kernighan's algorithm
    temp_knights = knights_board
    while temp_knights:
        # Get the square of the current knight
        from_square = (temp_knights & -temp_knights).bit_length() - 1
        
        # Get possible jumps from this square (including off the board, onto friendlies)
        attacks = KNIGHT_ATTACK_TABLE[from_square]
        
        # Filter out friendly fire
        valid_destinations = attacks & ~friendlies
        
        # Extract the individual destination squares
        temp_destinations = valid_destinations
        while temp_destinations:
            to_square = (temp_destinations & -temp_destinations).bit_length() - 1
            
            # Check if this destination holds an enemy piece (Capture) or is empty (Quiet)
            flag = FLAG_CAPTURE if (all_pieces & (1 << to_square)) else FLAG_QUIET
            
            # Pack it into our 16-bit move integer and append
            moves.append(encode_move(from_square, to_square, flag))
            
            # Clear the least significant bit to move to the next destination
            temp_destinations &= temp_destinations - 1
            
        # Clear the least significant bit to move to the next knight
        temp_knights &= temp_knights - 1
        
    return moves

def get_bishop_moves(bishops_bb: int, friendlies: int, all_pieces: int) -> List[int]:
    moves = []
    temp = bishops_bb

    while temp:
        from_square = (temp & -temp).bit_length() - 1
        attacks = get_bishop_attacks(from_square, all_pieces)

        valid = attacks & ~friendlies

        temp_dest = valid
        while temp_dest:
            to_square = (temp_dest & -temp_dest).bit_length() - 1
            flag = FLAG_CAPTURE if (all_pieces & (1 << to_square)) else FLAG_QUIET
            moves.append(encode_move(from_square, to_square, flag))
            temp_dest &= temp_dest - 1

        temp &= temp - 1

    return moves

def get_rook_moves(rooks_bb: int, friendlies: int, all_pieces: int) -> List[int]:
    moves = []
    temp = rooks_bb

    while temp:
        from_square = (temp & -temp).bit_length() - 1
        attacks = get_rook_attacks(from_square, all_pieces)

        valid = attacks & ~friendlies

        temp_dest = valid
        while temp_dest:
            to_square = (temp_dest & -temp_dest).bit_length() - 1
            flag = FLAG_CAPTURE if (all_pieces & (1 << to_square)) else FLAG_QUIET
            moves.append(encode_move(from_square, to_square, flag))
            temp_dest &= temp_dest - 1

        temp &= temp - 1

    return moves

def get_queen_moves(queen_bb: int, friendlies: int, all_pieces: int) -> List[int]:

    moves = get_bishop_moves(queen_bb, friendlies, all_pieces)
    moves.extend(get_rook_moves(queen_bb, friendlies, all_pieces))

    return moves

def get_king_moves(king_board: int, friendlies: int, all_pieces: int) -> List[int]:
    moves = []
    if not king_board:
        return moves
    
    from_square = (king_board & -king_board).bit_length() - 1

    attacks = KING_ATTACK_TABLE[from_square]
    valid_destinations = attacks & ~friendlies

    temp = valid_destinations
    while temp:
        to_square = (temp & -temp).bit_length() - 1
        
        flag = FLAG_CAPTURE if (all_pieces & (1 << to_square)) else FLAG_QUIET
        moves.append(encode_move(from_square, to_square, flag))
        
        temp &= temp - 1
        
    return moves

def get_white_castling(board: Board) -> List[int]:
    moves = []

    if board.white_king & (1 << E1):
        if board.castling_rights & WHITE_SHORT_RIGHT:
            if (board.all_pieces & WHITE_SHORT_EMPTY_MASK) == 0:
                if not is_square_attacked(board, E1, attacked_by_white=False) and \
                not is_square_attacked(board, F1, attacked_by_white=False) and \
                not is_square_attacked(board, G1, attacked_by_white=False):
                    
                    moves.append(encode_move(E1, G1, FLAG_SHORT_CASTLE))
                    
        if board.castling_rights & WHITE_LONG_RIGHT:
            if (board.all_pieces & WHITE_LONG_EMPTY_MASK) == 0:
                if not is_square_attacked(board, C1, attacked_by_white=False) and \
                not is_square_attacked(board, D1, attacked_by_white=False) and \
                not is_square_attacked(board, E1, attacked_by_white=False):
                    
                    moves.append(encode_move(E1, C1, FLAG_LONG_CASTLE))
                
    return moves

def get_black_castling(board: Board) -> List[int]:
    moves = []

    if board.black_king & (1 << E8):
        if board.castling_rights & BLACK_SHORT_RIGHT:
            if (board.all_pieces & BLACK_SHORT_EMPTY_MASK) == 0:
                # Black is checking if WHITE is attacking these squares
                if not is_square_attacked(board, E8, attacked_by_white=True) and \
                not is_square_attacked(board, F8, attacked_by_white=True) and \
                not is_square_attacked(board, G8, attacked_by_white=True):
                    
                    moves.append(encode_move(E8, G8, FLAG_SHORT_CASTLE))

        if board.castling_rights & BLACK_LONG_RIGHT:
            if (board.all_pieces & BLACK_LONG_EMPTY_MASK) == 0:
                if not is_square_attacked(board, C8, attacked_by_white=True) and \
                not is_square_attacked(board, D8, attacked_by_white=True) and \
                not is_square_attacked(board, E8, attacked_by_white=True):
                    
                    moves.append(encode_move(E8, C8, FLAG_LONG_CASTLE))
                
    return moves

def is_square_attacked(board: Board, square: int, attacked_by_white: bool = False) -> bool:

    if attacked_by_white:
        pawns   = board.white_pawns
        knights = board.white_knights
        bishops = board.white_bishops
        rooks   = board.white_rooks
        queens  = board.white_queens
        king    = board.white_king
    else:
        pawns   = board.black_pawns
        knights = board.black_knights
        bishops = board.black_bishops
        rooks   = board.black_rooks
        queens  = board.black_queens
        king    = board.black_king
    
    pawn_attacks = BLACK_PAWN_ATTACK_TABLE[square] if attacked_by_white else WHITE_PAWN_ATTACK_TABLE[square]
    if pawn_attacks & pawns:
        return True
    
    if KNIGHT_ATTACK_TABLE[square] & knights:
        return True
    
    if KING_ATTACK_TABLE[square] & king:
        return True
    
    straight_attacks = get_rook_attacks(square, board.all_pieces)
    if straight_attacks & (rooks | queens):
        return True

    diagonal_attacks = get_bishop_attacks(square, board.all_pieces)
    if diagonal_attacks & (bishops | queens):
        return True

    return False

def get_pseudo_legal_moves(board: Board) -> List[int]:
    moves = []
    
    if board.white_to_move:
        friendlies = board.white_pieces
        knights    = board.white_knights
        bishops    = board.white_bishops
        rooks      = board.white_rooks
        queens     = board.white_queens
        king       = board.white_king
        
        # Process the Asymmetric White logic
        moves.extend(get_white_pawn_moves(board))
        moves.extend(get_white_castling(board))
    else:
        friendlies = board.black_pieces
        knights    = board.black_knights
        bishops    = board.black_bishops
        rooks      = board.black_rooks
        queens     = board.black_queens
        king       = board.black_king
        
        moves.extend(get_black_pawn_moves(board))
        moves.extend(get_black_castling(board))
    
    # Static Lookups
    moves.extend(get_knight_moves(knights, friendlies, board.all_pieces))
    moves.extend(get_king_moves(king, friendlies, board.all_pieces)) # Normal 1-square moves
    
    # Magic Bitboard Lookups
    moves.extend(get_bishop_moves(bishops, friendlies, board.all_pieces))
    moves.extend(get_rook_moves(rooks, friendlies, board.all_pieces))
    moves.extend(get_queen_moves(queens, friendlies, board.all_pieces))

    return moves

def get_all_legal_moves(board: Board) -> List[int]:
    pseudo_legal_moves = get_pseudo_legal_moves(board)
    legal_moves = []

    enemy_is_white = not board.white_to_move

    for move in pseudo_legal_moves:
        board.make_move(move)

        king_board = board.black_king if board.white_to_move else board.white_king
        king_sq = (king_board & -king_board).bit_length() - 1

        if not is_square_attacked(board, king_sq, attacked_by_white=enemy_is_white):
            legal_moves.append(move)
        
        board.unmake_move(move)
    
    return legal_moves