"""
    Program: bitboard.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: Define the chessboard itself through the Board class. Use bitshifts for updates for
             efficiency.
"""

from core.constants import RANK_2, RANK_7
from core.constants import A1, B1, C1, D1, E1, F1, G1, H1, A8, B8, C8, D8, E8, F8, G8, H8
from core.constants import get_from_square, get_to_square, get_flag
from core.constants import (
    FLAG_CAPTURE, FLAG_DOUBLE_PUSH, FLAG_EP_CAPTURE, FLAG_LONG_CASTLE, FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_BISHOP_CAPTURE,
    FLAG_PROMOTE_KNIGHT, FLAG_PROMOTE_KNIGHT_CAPTURE, FLAG_PROMOTE_QUEEN, FLAG_PROMOTE_QUEEN_CAPTURE, FLAG_PROMOTE_ROOK,
    FLAG_PROMOTE_ROOK_CAPTURE, FLAG_QUIET, FLAG_SHORT_CASTLE,
    WHITE_SHORT_RIGHT, WHITE_LONG_RIGHT, BLACK_SHORT_RIGHT, BLACK_LONG_RIGHT
)
from core.zobrist import compute_hash, update_hash

"""
Representing Squares:
Each square has its own integer. Activating a square (a piece being on said square) can be done by "calling" 2^{square int}.
To flip D6 to 1, we use its representation, 2^{43} = 0x80000000000 (0x meaning hexidecimal, 80000000000 being the hex).
"""

class Board:
    def __init__(self):
        # White pieces initialization
        self.white_pawns   = 0
        self.white_knights = 0
        self.white_bishops = 0
        self.white_rooks   = 0
        self.white_queens  = 0
        self.white_king    = 0

        # Black pieces initialization
        self.black_pawns   = 0
        self.black_knights = 0
        self.black_bishops = 0
        self.black_rooks   = 0
        self.black_queens  = 0
        self.black_king    = 0

        # Piece sets initialization
        self.white_pieces  = 0
        self.black_pieces  = 0
        self.all_pieces    = 0

        self.white_to_move = True
        self.history = []
        self.en_passant_square = None
        self.castling_rights = 0b1111
        self.repetition_table = {}

        self.zobrist_hash = 0
    
    def set_starting_position(self):
        """
        Initialize the bitboard with Chess starting position
        """
        self.white_pawns   = RANK_2
        self.white_knights = (1 << B1) | (1 << G1)
        self.white_bishops = (1 << C1) | (1 << F1)
        self.white_rooks   = (1 << A1) | (1 << H1)
        self.white_queens  = (1 << D1)
        self.white_king    = (1 << E1)

        self.black_pawns   = RANK_7
        self.black_knights = (1 << B8) | (1 << G8)
        self.black_bishops = (1 << C8) | (1 << F8)
        self.black_rooks   = (1 << A8) | (1 << H8)
        self.black_queens  = (1 << D8)
        self.black_king    = (1 << E8)

        self.update_summary_boards()
        self.zobrist_hash = compute_hash(self)
        self.repetition_table[self.zobrist_hash] = 1

    def update_summary_boards(self):
        """
        Update pieces bitwise.
        """
        self.white_pieces = (self.white_pawns   | self.white_knights |
                             self.white_bishops | self.white_rooks   |
                             self.white_queens  | self.white_king     )
        
        self.black_pieces = (self.black_pawns   | self.black_knights |
                             self.black_bishops | self.black_rooks   |
                             self.black_queens  | self.black_king     )
        
        self.all_pieces = self.white_pieces | self.black_pieces
    
    def get_piece_at(self, square: int):
        """
        Returns a (color, piece_type) tuple for the piece on this square,
        or None if empty. e.g. ('white', 'rook'), ('black', 'pawn')
        """

        bit = 1 << square
        if self.white_pawns   & bit: return ('white', 'pawns')
        if self.white_knights & bit: return ('white', 'knights')
        if self.white_bishops & bit: return ('white', 'bishops')
        if self.white_rooks   & bit: return ('white', 'rooks')
        if self.white_queens  & bit: return ('white', 'queens')
        if self.white_king    & bit: return ('white', 'king')
        if self.black_pawns   & bit: return ('black', 'pawns')
        if self.black_knights & bit: return ('black', 'knights')
        if self.black_bishops & bit: return ('black', 'bishops')
        if self.black_rooks   & bit: return ('black', 'rooks')
        if self.black_queens  & bit: return ('black', 'queens')
        if self.black_king    & bit: return ('black', 'king')

        return None
    
    def set_bit(self, bitboard, square):
        return bitboard | (1 << square)
    
    def clear_bit(self, bitboard, square):
        return bitboard & ~(1 << square)
    
    def get_bit(self, bitboard, square):
        return bitboard & (1 << square)
    
    # ====================================================
    # make and unmake moves
    # ====================================================
    def make_move(self, move: int):
            from_sq = get_from_square(move)
            to_sq   = get_to_square(move)
            flag    = get_flag(move)

            # ====================================================
            # Save snapshot to history stack before changing anything
            # ====================================================
            snapshot = {
                'white_pawns':   self.white_pawns,
                'white_knights': self.white_knights,
                'white_bishops': self.white_bishops,
                'white_rooks':   self.white_rooks,
                'white_queens':  self.white_queens,
                'white_king':    self.white_king,
                'black_pawns':   self.black_pawns,
                'black_knights': self.black_knights,
                'black_bishops': self.black_bishops,
                'black_rooks':   self.black_rooks,
                'black_queens':  self.black_queens,
                'black_king':    self.black_king,
                'castling_rights':    self.castling_rights,
                'en_passant_square':  self.en_passant_square,
                'white_to_move':      self.white_to_move,
                'zobrist_hash':  self.zobrist_hash,
            }
            self.history.append(snapshot)

            self.zobrist_hash = update_hash(self.zobrist_hash, move, self)

            self.repetition_table[self.zobrist_hash] = self.repetition_table.get(self.zobrist_hash, 0) + 1

            color, piece = self.get_piece_at(from_sq)
            moving_bb    = f"{color}_{piece}"  # e.g. "white_rooks"

            # Clear en passant square
            self.en_passant_square = None

            if flag == FLAG_QUIET:
                setattr(self, moving_bb, self.clear_bit(getattr(self, moving_bb), from_sq))
                setattr(self, moving_bb, self.set_bit(getattr(self, moving_bb), to_sq))

            elif flag == FLAG_DOUBLE_PUSH:
                setattr(self, moving_bb, self.clear_bit(getattr(self, moving_bb), from_sq))
                setattr(self, moving_bb, self.set_bit(getattr(self, moving_bb), to_sq))
                # Set en passant square to the square the pawn skipped over
                self.en_passant_square = to_sq - 8 if color == 'white' else to_sq + 8

            elif flag == FLAG_CAPTURE:
                # Remove the captured piece first
                captured = self.get_piece_at(to_sq)
                if captured:
                    cap_color, cap_piece = captured
                    cap_bb = f"{cap_color}_{cap_piece}"
                    setattr(self, cap_bb, self.clear_bit(getattr(self, cap_bb), to_sq))
                # Move the attacker
                setattr(self, moving_bb, self.clear_bit(getattr(self, moving_bb), from_sq))
                setattr(self, moving_bb, self.set_bit(getattr(self, moving_bb), to_sq))

            elif flag == FLAG_EP_CAPTURE:
                # Move the attacking pawn
                setattr(self, moving_bb, self.clear_bit(getattr(self, moving_bb), from_sq))
                setattr(self, moving_bb, self.set_bit(getattr(self, moving_bb), to_sq))
                # Remove the captured pawn (one rank behind to_sq)
                captured_pawn_sq = to_sq - 8 if color == 'white' else to_sq + 8
                if color == 'white':
                    self.black_pawns = self.clear_bit(self.black_pawns, captured_pawn_sq)
                else:
                    self.white_pawns = self.clear_bit(self.white_pawns, captured_pawn_sq)

            elif flag == FLAG_SHORT_CASTLE:
                if color == 'white':
                    self.white_king  = self.clear_bit(self.white_king,  E1)
                    self.white_king  = self.set_bit(self.white_king,    G1)
                    self.white_rooks = self.clear_bit(self.white_rooks, H1)
                    self.white_rooks = self.set_bit(self.white_rooks,   F1)
                else:
                    self.black_king  = self.clear_bit(self.black_king,  E8)
                    self.black_king  = self.set_bit(self.black_king,    G8)
                    self.black_rooks = self.clear_bit(self.black_rooks, H8)
                    self.black_rooks = self.set_bit(self.black_rooks,   F8)

            elif flag == FLAG_LONG_CASTLE:
                if color == 'white':
                    self.white_king  = self.clear_bit(self.white_king,  E1)
                    self.white_king  = self.set_bit(self.white_king,    C1)
                    self.white_rooks = self.clear_bit(self.white_rooks, A1)
                    self.white_rooks = self.set_bit(self.white_rooks,   D1)
                else:
                    self.black_king  = self.clear_bit(self.black_king,  E8)
                    self.black_king  = self.set_bit(self.black_king,    C8)
                    self.black_rooks = self.clear_bit(self.black_rooks, A8)
                    self.black_rooks = self.set_bit(self.black_rooks,   D8)

            elif flag in (FLAG_PROMOTE_QUEEN,  FLAG_PROMOTE_ROOK,
                        FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_KNIGHT,
                        FLAG_PROMOTE_QUEEN_CAPTURE,  FLAG_PROMOTE_ROOK_CAPTURE,
                        FLAG_PROMOTE_BISHOP_CAPTURE, FLAG_PROMOTE_KNIGHT_CAPTURE):

                # Remove captured piece first if this is a capture-promotion
                if flag >= FLAG_PROMOTE_KNIGHT_CAPTURE:
                    captured = self.get_piece_at(to_sq)
                    if captured:
                        cap_color, cap_piece = captured
                        cap_bb = f"{cap_color}_{cap_piece}"
                        setattr(self, cap_bb, self.clear_bit(getattr(self, cap_bb), to_sq))

                # Remove the pawn
                setattr(self, moving_bb, self.clear_bit(getattr(self, moving_bb), from_sq))

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
                promoted_bb = promote_map[flag]
                setattr(self, promoted_bb, self.set_bit(getattr(self, promoted_bb), to_sq))

            # Update castling rights
            if from_sq == E1 or to_sq == E1: self.castling_rights &= ~WHITE_SHORT_RIGHT & ~WHITE_LONG_RIGHT
            if from_sq == H1 or to_sq == H1: self.castling_rights &= ~WHITE_SHORT_RIGHT
            if from_sq == A1 or to_sq == A1: self.castling_rights &= ~WHITE_LONG_RIGHT
            if from_sq == E8 or to_sq == E8: self.castling_rights &= ~BLACK_SHORT_RIGHT & ~BLACK_LONG_RIGHT
            if from_sq == H8 or to_sq == H8: self.castling_rights &= ~BLACK_SHORT_RIGHT
            if from_sq == A8 or to_sq == A8: self.castling_rights &= ~BLACK_LONG_RIGHT

            # Flip the turn and update summary boards
            self.white_to_move = not self.white_to_move
            self.update_summary_boards()

    def unmake_move(self, move: int):
        snapshot = self.history.pop()

        post_move_hash = self.zobrist_hash
        self.repetition_table[post_move_hash] -= 1
        if self.repetition_table[post_move_hash] == 0:
            del self.repetition_table[post_move_hash]

        self.white_pawns   = snapshot['white_pawns']
        self.white_knights = snapshot['white_knights']
        self.white_bishops = snapshot['white_bishops']
        self.white_rooks   = snapshot['white_rooks']
        self.white_queens  = snapshot['white_queens']
        self.white_king    = snapshot['white_king']
        self.black_pawns   = snapshot['black_pawns']
        self.black_knights = snapshot['black_knights']
        self.black_bishops = snapshot['black_bishops']
        self.black_rooks   = snapshot['black_rooks']
        self.black_queens  = snapshot['black_queens']
        self.black_king    = snapshot['black_king']

        self.castling_rights   = snapshot['castling_rights']
        self.en_passant_square = snapshot['en_passant_square']
        self.white_to_move     = snapshot['white_to_move']

        self.zobrist_hash  = snapshot['zobrist_hash']

        self.update_summary_boards()
    
    def load_fen(self, fen: str):
        """
        Loads a position from a FEN string.
        FEN format: piece_placement side_to_move castling ep_square halfmove fullmove
        Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
        """
        # Reset all bitboards
        self.white_pawns = self.white_knights = self.white_bishops = 0
        self.white_rooks = self.white_queens  = self.white_king    = 0
        self.black_pawns = self.black_knights = self.black_bishops = 0
        self.black_rooks = self.black_queens  = self.black_king    = 0
        self.castling_rights   = 0
        self.en_passant_square = None
        self.history           = []
        self.repetition_table  = {}

        parts = fen.strip().split()
        piece_placement = parts[0]
        side_to_move    = parts[1]
        castling        = parts[2]
        ep_square       = parts[3]

        # ====================================================
        # Piece placement
        # FEN ranks go from rank 8 (top) to rank 1 (bottom)
        # ====================================================
        piece_map = {
            'P': 'white_pawns',   'p': 'black_pawns',
            'N': 'white_knights', 'n': 'black_knights',
            'B': 'white_bishops', 'b': 'black_bishops',
            'R': 'white_rooks',   'r': 'black_rooks',
            'Q': 'white_queens',  'q': 'black_queens',
            'K': 'white_king',    'k': 'black_king',
        }

        rank = 7  # start at rank 8 (index 7)
        file = 0

        for char in piece_placement:
            if char == '/':
                rank -= 1
                file  = 0
            elif char.isdigit():
                file += int(char)
            else:
                square = rank * 8 + file
                attr   = piece_map[char]
                setattr(self, attr, getattr(self, attr) | (1 << square))
                file += 1

        # ====================================================
        # Side to move
        # ====================================================
        self.white_to_move = (side_to_move == 'w')

        # ====================================================
        # Castling rights
        # ====================================================
        if 'K' in castling: self.castling_rights |= WHITE_SHORT_RIGHT
        if 'Q' in castling: self.castling_rights |= WHITE_LONG_RIGHT
        if 'k' in castling: self.castling_rights |= BLACK_SHORT_RIGHT
        if 'q' in castling: self.castling_rights |= BLACK_LONG_RIGHT

        # ====================================================
        # En passant square
        # ====================================================
        if ep_square != '-':
            file = ord(ep_square[0]) - ord('a')
            rank = int(ep_square[1]) - 1
            self.en_passant_square = rank * 8 + file

        # ====================================================
        # Update summary boards and Zobrist hash
        # ====================================================
        self.update_summary_boards()
        self.zobrist_hash = compute_hash(self)
        self.repetition_table[self.zobrist_hash] = 1