# Define Chess Board: assign int to every square (0-63)
A1, B1, C1, D1, E1, F1, G1, H1 = range(8)
A2, B2, C2, D2, E2, F2, G2, H2 = range(8,16)
A3, B3, C3, D3, E3, F3, G3, H3 = range(16,24)
A4, B4, C4, D4, E4, F4, G4, H4 = range(24,32)
A5, B5, C5, D5, E5, F5, G5, H5 = range(32,40)
A6, B6, C6, D6, E6, F6, G6, H6 = range(40,48)
A7, B7, C7, D7, E7, F7, G7, H7 = range(48,56)
A8, B8, C8, D8, E8, F8, G8, H8 = range(56,64)

# Define ranks (rows)
RANK_1 = 0x00000000000000FF
RANK_2 = 0x000000000000FF00
RANK_3 = 0x0000000000FF0000
RANK_4 = 0x00000000FF000000
RANK_5 = 0x000000FF00000000
RANK_6 = 0x0000FF0000000000
RANK_7 = 0x00FF000000000000
RANK_8 = 0xFF00000000000000

# Define Files (columns)
FILE_A = 0x0101010101010101
FILE_B = 0x0202020202020202
FILE_C = 0x0404040404040404
FILE_D = 0x0808080808080808
FILE_E = 0x1010101010101010
FILE_F = 0x2020202020202020
FILE_G = 0x4040404040404040
FILE_H = 0x8080808080808080

# Square colors and ensuring pieces don't jump into the void and die horrifically
LIGHT_SQUARES = 0x55AA55AA55AA55AA
DARK_SQUARES  = 0xAA55AA55AA55AA55
MASK_64 = 0xFFFFFFFFFFFFFFFF

# Castling
WHITE_SHORT_EMPTY_MASK  = (1 << F1) | (1 << G1)
WHITE_LONG_EMPTY_MASK   = (1 << B1) | (1 << C1) | (1 << D1)
BLACK_SHORT_EMPTY_MASK  = (1 << F8) | (1 << G8)
BLACK_LONG_EMPTY_MASK   = (1 << B8) | (1 << C8) | (1 << D8)

WHITE_SHORT_RIGHT = 0b0001
WHITE_LONG_RIGHT  = 0b0010
BLACK_SHORT_RIGHT = 0b0100
BLACK_LONG_RIGHT  = 0b1000

# Flags
FLAG_QUIET                  = 0
FLAG_DOUBLE_PUSH            = 1
FLAG_SHORT_CASTLE           = 2
FLAG_LONG_CASTLE            = 3
FLAG_CAPTURE                = 4
FLAG_EP_CAPTURE             = 5
FLAG_PROMOTE_KNIGHT         = 8
FLAG_PROMOTE_BISHOP         = 9
FLAG_PROMOTE_ROOK           = 10
FLAG_PROMOTE_QUEEN          = 11
FLAG_PROMOTE_KNIGHT_CAPTURE = 12
FLAG_PROMOTE_BISHOP_CAPTURE = 13
FLAG_PROMOTE_ROOK_CAPTURE   = 14
FLAG_PROMOTE_QUEEN_CAPTURE  = 15

def encode_move(from_sq: int, to_sq: int, flag=FLAG_QUIET) -> int:
    return from_sq | (to_sq << 6) | (flag << 12)

def get_from_square(move: int) -> int:
    return move & 0x3F # First 6 bits

def get_to_square(move: int) -> int:
    return (move >> 6) & 0x3F

def get_flag(move: int) -> int:
    return (move >> 12) & 0x0F