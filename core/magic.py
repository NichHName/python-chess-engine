"""
    Program: magic.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: Magic bitboard precomputation and runtime attack lookup for 
             sliding pieces (rook, bishop). All precomputation functions 
             are private (_prefix) and run once at startup. Only 
             get_rook_attacks and get_bishop_attacks are public.
"""

from typing import List, NamedTuple

# =========================================================
# Magic Number Logic
# =========================================================
ROOK_MAGIC_NUMBERS = [
    0x0080001020400080, 0x0040001000200040, 0x0080081000200080, 0x0080040800100080,
    0x0080020400080080, 0x0080010200040080, 0x0080008001000200, 0x0080002040800100,
    0x0000800020400080, 0x0000400020005000, 0x0000801000200080, 0x0000800800100080,
    0x0000800400080080, 0x0000800200040080, 0x0000800100020080, 0x0000800040800100,
    0x0000208000400080, 0x0000404000201000, 0x0000808010002000, 0x0000808008001000,
    0x0000808004000800, 0x0000808002000400, 0x0000010100020004, 0x0000020000408104,
    0x0000208080004000, 0x0000200040005000, 0x0000100080200080, 0x0000080080100080,
    0x0000040080080080, 0x0000020080040080, 0x0000010080800200, 0x0000800080004100,
    0x0000204000800080, 0x0000200040401000, 0x0000100080802000, 0x0000080080801000,
    0x0000040080800800, 0x0000020080800400, 0x0000020001010004, 0x0000800040800100,
    0x0000204000808000, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
    0x0000040008008080, 0x0000020004008080, 0x0000010002008080, 0x0000004081020004,
    0x0000204000800080, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
    0x0000040008008080, 0x0000020004008080, 0x0000800100020080, 0x0000800041000080,
    0x00FFFCDDFCED714A, 0x007FFCDDFCED714A, 0x003FFFCDFFD88096, 0x0000040810002101,
    0x0001000204080011, 0x0001000204000801, 0x0001000082000401, 0x0001FFFAABFAD1A2,
]

BISHOP_MAGIC_NUMBERS = [
    0x0002020202020200, 0x0002020202020000, 0x0004010202000000, 0x0004040080000000,
    0x0001104000000000, 0x0000821040000000, 0x0000410410400000, 0x0000104104104000,
    0x0000040404040400, 0x0000020202020200, 0x0000040102020000, 0x0000040400800000,
    0x0000011040000000, 0x0000008210400000, 0x0000004104104000, 0x0000002082082000,
    0x0004000808080800, 0x0002000404040400, 0x0001000202020200, 0x0000800802004000,
    0x0000800400A00000, 0x0000200100884000, 0x0000400082082000, 0x0000200041041000,
    0x0002080010101000, 0x0001040008080800, 0x0000208004010400, 0x0000404004010200,
    0x0000840000802000, 0x0000404002011000, 0x0000808001041000, 0x0000404000820800,
    0x0001041000202000, 0x0000820800101000, 0x0000104400080800, 0x0000020080080080,
    0x0000404040040100, 0x0000808100020100, 0x0001010100020800, 0x0000808080010400,
    0x0000820820004000, 0x0000410410002000, 0x0000082088001000, 0x0000002011000800,
    0x0000080100400400, 0x0001010101000200, 0x0002020202000400, 0x0001010101000200,
    0x0000410410400000, 0x0000208208200000, 0x0000002084000000, 0x0000000020880000,
    0x0000001002020000, 0x0000040408020000, 0x0004040404040000, 0x0002020202020000,
    0x0000104104104000, 0x0000002082082000, 0x0000000020841000, 0x0000000008220400,
    0x0000000002088200, 0x0000000000842100, 0x0000000000209440, 0x0000000000082310,
]

class SMagic(NamedTuple):
    offset: int   # index into ATTACK_TABLE where this square's attacks start
    mask:   int
    magic:  int
    shift:  int

MASK_64 = 0xFFFFFFFFFFFFFFFF
ATTACK_TABLE: List[int] = [0] * 102400
ROOK_TABLE:   List[SMagic] = [None] * 64
BISHOP_TABLE: List[SMagic] = [None] * 64

# Private Helper
def _compute_bishop_blocker_mask(square: int) -> int:
    """
    The squares whose occupancy matters for a bishop on this square.
    Edges excluded for the same reason as the rook.
    """
    mask = 0
    rank, file = square // 8, square % 8

    r, f = rank + 1, file + 1
    while r < 7 and f < 7: mask |= 1 << (r * 8 + f); r += 1; f += 1  # NE

    r, f = rank + 1, file - 1
    while r < 7 and f > 0: mask |= 1 << (r * 8 + f); r += 1; f -= 1  # NW

    r, f = rank - 1, file + 1
    while r > 0 and f < 7: mask |= 1 << (r * 8 + f); r -= 1; f += 1  # SE

    r, f = rank - 1, file - 1
    while r > 0 and f > 0: mask |= 1 << (r * 8 + f); r -= 1; f -= 1  # SW

    return mask

# Private Helper
def _compute_rook_blocker_mask(square: int) -> int:
    """
    The squares whose occupancy matters for a rook on this square.
    Edges are excluded: an edge piece is always captured or not, never a blocker.
    """
    mask = 0
    rank, file = square // 8, square % 8

    for r in range(rank + 1, 7): mask |= 1 << (r * 8 + file)   # north, stop before rank 8
    for r in range(rank - 1, 0, -1): mask |= 1 << (r * 8 + file) # south, stop before rank 1
    for f in range(file + 1, 7): mask |= 1 << (rank * 8 + f)   # east,  stop before file H
    for f in range(file - 1, 0, -1): mask |= 1 << (rank * 8 + f) # west,  stop before file A

    return mask

# Private Helper
def _compute_bishop_attacks_slow(square: int, occupancy: int) -> int:
    """
    Ground truth bishop attacks given a specific occupancy.
    Never called at runtime — only during precomputation.
    """
    attacks = 0
    rank, file = square // 8, square % 8

    r, f = rank + 1, file + 1
    while r < 8 and f < 8:
        attacks |= 1 << (r * 8 + f)
        if occupancy & (1 << (r * 8 + f)): break
        r += 1; f += 1

    r, f = rank + 1, file - 1
    while r < 8 and f >= 0:
        attacks |= 1 << (r * 8 + f)
        if occupancy & (1 << (r * 8 + f)): break
        r += 1; f -= 1

    r, f = rank - 1, file + 1
    while r >= 0 and f < 8:
        attacks |= 1 << (r * 8 + f)
        if occupancy & (1 << (r * 8 + f)): break
        r -= 1; f += 1

    r, f = rank - 1, file - 1
    while r >= 0 and f >= 0:
        attacks |= 1 << (r * 8 + f)
        if occupancy & (1 << (r * 8 + f)): break
        r -= 1; f -= 1

    return attacks

# Private Helper
def _compute_rook_attacks_slow(square: int, occupancy: int) -> int:
    """
    Ground truth rook attacks given a specific occupancy.
    Slides in each direction, stops when hitting an occupied square (inclusive).
    Never called at runtime — only during precomputation.
    """
    attacks = 0
    rank, file = square // 8, square % 8

    for r in range(rank + 1, 8):
        attacks |= 1 << (r * 8 + file)
        if occupancy & (1 << (r * 8 + file)): break

    for r in range(rank - 1, -1, -1):
        attacks |= 1 << (r * 8 + file)
        if occupancy & (1 << (r * 8 + file)): break

    for f in range(file + 1, 8):
        attacks |= 1 << (rank * 8 + f)
        if occupancy & (1 << (rank * 8 + f)): break

    for f in range(file - 1, -1, -1):
        attacks |= 1 << (rank * 8 + f)
        if occupancy & (1 << (rank * 8 + f)): break

    return attacks

def _index_to_occupancy(index: int, mask: int) -> int:
    """
    Maps an integer index to a unique subset of the bits in mask.
    Uses the carry-rippler technique to enumerate all 2^n subsets.
    """
    occupancy = 0
    bits = mask
    for i in range(bin(mask).count('1')):
        lsb   = bits & -bits      # isolate lowest set bit of mask
        bits &= bits - 1          # clear it from our working copy
        if index & (1 << i):      # if this bit is 'on' in the index
            occupancy |= lsb      # include the corresponding square
    return occupancy

def precompute_bishop_magic_table() -> None:
    # Bishop slices are placed after all rook slices in the flat table
    offset = sum(1 << bin(_compute_rook_blocker_mask(sq)).count('1') for sq in range(64))

    for square in range(64):
        mask     = _compute_bishop_blocker_mask(square)
        num_bits = bin(mask).count('1')
        shift    = 64 - num_bits
        size     = 1 << num_bits

        BISHOP_TABLE[square] = SMagic(
            offset = offset,
            mask   = mask,
            magic  = BISHOP_MAGIC_NUMBERS[square],
            shift  = shift
        )

        for i in range(size):
            occ     = _index_to_occupancy(i, mask)
            attacks = _compute_bishop_attacks_slow(square, occ)
            index   = (occ * BISHOP_MAGIC_NUMBERS[square] & MASK_64) >> shift
            ATTACK_TABLE[offset + index] = attacks

        offset += size

def precompute_rook_magic_table() -> None:
    offset = 0
    for square in range(64):
        mask     = _compute_rook_blocker_mask(square)
        num_bits = bin(mask).count('1')
        shift    = 64 - num_bits
        size     = 1 << num_bits

        ROOK_TABLE[square] = SMagic(
            offset = offset,
            mask   = mask,
            magic  = ROOK_MAGIC_NUMBERS[square],
            shift  = shift
        )

        for i in range(size):
            occ     = _index_to_occupancy(i, mask)
            attacks = _compute_rook_attacks_slow(square, occ)
            index   = (occ * ROOK_MAGIC_NUMBERS[square] & MASK_64) >> shift
            ATTACK_TABLE[offset + index] = attacks

        offset += size

def get_bishop_attacks(square: int, occupancy: int) -> int:
    entry  = BISHOP_TABLE[square]
    occ    = (occupancy & entry.mask) * entry.magic & MASK_64
    return ATTACK_TABLE[entry.offset + (occ >> entry.shift)]

def get_rook_attacks(square: int, occupancy: int) -> int:
    entry  = ROOK_TABLE[square]
    occ    = (occupancy & entry.mask) * entry.magic & MASK_64
    return ATTACK_TABLE[entry.offset + (occ >> entry.shift)]

precompute_rook_magic_table()
precompute_bishop_magic_table()