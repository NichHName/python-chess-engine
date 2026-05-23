"""
    Program: perft.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: Move generation verification via perft (performance test).
             Counts leaf nodes at a given depth and compares against
             known correct values to verify move generation correctness.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.bitboard import Board
from core.move_gen import get_all_legal_moves
from core.constants import get_from_square, get_to_square, get_flag
from core.zobrist import compute_hash

# =========================================================
# Known correct perft values from starting position
# =========================================================
PERFT_EXPECTED = {
    1: 20,
    2: 400,
    3: 8902,
    4: 197281,
    5: 4865609,
}

# =========================================================
# Core perft functions
# =========================================================
def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1

    moves = get_all_legal_moves(board)
    nodes = 0

    for move in moves:
        board.make_move(move)
        nodes += perft(board, depth - 1)
        board.unmake_move(move)

    return nodes

def perft_divide(board: Board, depth: int) -> int:
    """
    Prints the node count for each root move.
    Invaluable for pinpointing which move is generating wrong counts.
    """
    moves = get_all_legal_moves(board)
    total = 0

    for move in moves:
        board.make_move(move)
        count = perft(board, depth - 1)
        board.unmake_move(move)

        from_sq = get_from_square(move)
        to_sq   = get_to_square(move)
        print(f"{_square_to_algebraic(from_sq)}{_square_to_algebraic(to_sq)}: {count}")
        total += count

    print(f"\nTotal: {total}")
    return total

# =========================================================
# Verification against known values
# =========================================================
def verify_perft(max_depth: int = 4):
    """
    Runs perft from the starting position and checks against
    known correct values. Prints pass/fail for each depth.
    """
    board = Board()
    board.set_starting_position()

    print("Verifying perft from starting position...\n")
    all_pass = True

    for depth in range(1, max_depth + 1):
        result   = perft(board, depth)
        expected = PERFT_EXPECTED[depth]
        status   = "PASS" if result == expected else "FAIL"

        if status == "FAIL":
            all_pass = False

        print(f"  depth {depth}: {result:>10,}  expected value was {expected:>10,}  [{status}]")

    print()
    if all_pass:
        print("All perft tests passed — move generation is correct.")
    else:
        print("Failures detected — use perft_divide to isolate the bug.")

# =========================================================
# Helpers
# =========================================================
def _square_to_algebraic(square: int) -> str:
    file = 'abcdefgh'[square % 8]
    rank = str((square // 8) + 1)
    return file + rank

def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1

    moves = get_all_legal_moves(board)
    nodes = 0

    for move in moves:
        board.make_move(move)
        nodes += perft(board, depth - 1)
        board.unmake_move(move)
        
        # Verify hash consistency
        assert board.zobrist_hash == compute_hash(board), \
            f"Hash mismatch after unmake! move={move}, flag={get_flag(move)}"

    return nodes