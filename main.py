"""
    Program: main.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: Entry point. Runs the engine against itself and produces
             a PGN file that can be imported into any chess viewer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.bitboard import Board
from core.constants import (
    get_from_square, get_to_square, get_flag,
    FLAG_PROMOTE_QUEEN, FLAG_PROMOTE_ROOK,
    FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_KNIGHT,
    FLAG_PROMOTE_QUEEN_CAPTURE, FLAG_PROMOTE_ROOK_CAPTURE,
    FLAG_PROMOTE_BISHOP_CAPTURE, FLAG_PROMOTE_KNIGHT_CAPTURE,
)
from core.move_gen import get_all_legal_moves, is_square_attacked
from search.minimax import get_best_move

# ====================================================
# Configuration
# ====================================================
SEARCH_DEPTH = 3
MAX_MOVES    = 55   # prevent infinite games
OUTPUT_FILE  = "game.pgn"

# ====================================================
# Helpers
# ====================================================
def square_to_algebraic(square: int) -> str:
    file = 'abcdefgh'[square % 8]
    rank = str((square // 8) + 1)
    return file + rank

def move_to_uci(move: int) -> str:
    from_sq = get_from_square(move)
    to_sq   = get_to_square(move)
    flag    = get_flag(move)

    promotion_map = {
        FLAG_PROMOTE_QUEEN:          'q',
        FLAG_PROMOTE_ROOK:           'r',
        FLAG_PROMOTE_BISHOP:         'b',
        FLAG_PROMOTE_KNIGHT:         'n',
        FLAG_PROMOTE_QUEEN_CAPTURE:  'q',
        FLAG_PROMOTE_ROOK_CAPTURE:   'r',
        FLAG_PROMOTE_BISHOP_CAPTURE: 'b',
        FLAG_PROMOTE_KNIGHT_CAPTURE: 'n',
    }

    uci = square_to_algebraic(from_sq) + square_to_algebraic(to_sq)
    if flag in promotion_map:
        uci += promotion_map[flag]
    return uci

def write_pgn(moves_uci: list, result: str, filename: str):
    """
    Writes a PGN file from a list of UCI move strings.
    Uses a simple coordinate notation format (e.g. e2e4)
    which most viewers accept.
    """
    lines = []
    lines.append('[Event "Engine Self-Play"]')
    lines.append('[Site "Neural Chess Engine"]')
    lines.append('[Date "2026.05"]')
    lines.append('[White "Engine"]')
    lines.append('[Black "Engine"]')
    lines.append(f'[Result "{result}"]')
    lines.append('')

    # Format moves in PGN style: "1. e2e4 e7e5 2. ..."
    move_text = ''
    for i, uci in enumerate(moves_uci):
        if i % 2 == 0:
            move_text += f'{i // 2 + 1}. '
        move_text += uci + ' '

    move_text += result
    
    # Wrap at 80 characters
    words = move_text.split()
    line = ''
    for word in words:
        if len(line) + len(word) + 1 > 80:
            lines.append(line)
            line = word
        else:
            line = (line + ' ' + word).strip()
    if line:
        lines.append(line)

    with open(filename, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"PGN written to {filename}")

def self_play():
    board = Board()
    board.set_starting_position()

    moves_uci = []
    result    = '*'
    move_num  = 1

    print(f"Starting self-play at depth {SEARCH_DEPTH}...\n")

    for ply in range(MAX_MOVES * 2):
        legal_moves = get_all_legal_moves(board)

        if board.repetition_table.get(board.zobrist_hash, 0) >= 2:
            result = '1/2-1/2'
            print("Draw by repetition.")
            break

        # Terminal check
        if not legal_moves:
            from core.move_gen import is_square_attacked
            king = board.white_king if board.white_to_move else board.black_king
            king_sq = (king & -king).bit_length() - 1
            in_check = is_square_attacked(board, king_sq, attacked_by_white=not board.white_to_move)

            if in_check:
                result = '0-1' if board.white_to_move else '1-0'
                winner = "Black" if board.white_to_move else "White"
                print(f"Checkmate! {winner} wins.")
            else:
                result = '1/2-1/2'
                print("Stalemate! Draw.")
            break

        # Get best move
        move, score = get_best_move(board, SEARCH_DEPTH)

        if move is None:
            break

        uci = move_to_uci(move)
        moves_uci.append(uci)

        if board.white_to_move:
            print(f"{move_num}. {uci} (score: {score})", end='  ')
        else:
            print(f"{uci} (score: {score})")
            move_num += 1

        board.make_move(move)

    else:
        result = '1/2-1/2'
        print(f"\nDraw by move limit ({MAX_MOVES} moves).")

    print()
    write_pgn(moves_uci, result, OUTPUT_FILE)

def play_engine():
    board = Board()
    board.set_starting_position()

    print("Play against the engine!")
    print("Enter moves in UCI format (e.g. e2e4, g1f3, e1g1 for castling)")
    print("Enter 'quit' to exit\n")

    color = input("Play as white or black? (w/b): ").strip().lower()
    human_is_white = (color == 'w')

    move_num = 1

    while True:
        legal_moves = get_all_legal_moves(board)

        # Repetition check
        if board.repetition_table.get(board.zobrist_hash, 0) >= 2:
            print("Draw by repetition.")
            break

        # Terminal check
        if not legal_moves:
            king = board.white_king if board.white_to_move else board.black_king
            king_sq = (king & -king).bit_length() - 1
            in_check = is_square_attacked(board, king_sq, attacked_by_white=not board.white_to_move)
            if in_check:
                winner = "Black" if board.white_to_move else "White"
                print(f"Checkmate! {winner} wins.")
            else:
                print("Stalemate! Draw.")
            break

        human_to_move = (board.white_to_move == human_is_white)

        if human_to_move:
            # Print the board state
            side = "White" if board.white_to_move else "Black"
            print(f"\n{side} to move.")
            _print_legal_moves(board, legal_moves)

            while True:
                user_input = input("Your move: ").strip().lower()
                if user_input == 'quit':
                    return

                move = _parse_uci(user_input, legal_moves)
                if move is not None:
                    break
                print(f"Illegal move: {user_input}. Try again.")

            uci = move_to_uci(move)
            if board.white_to_move:
                print(f"{move_num}. {uci}", end='  ')
            else:
                print(f"{uci}")
                move_num += 1

            board.make_move(move)

        else:
            # Engine's turn
            side = "White" if board.white_to_move else "Black"
            print(f"\nEngine ({side}) thinking...")
            move, score = get_best_move(board, SEARCH_DEPTH)

            if move is None:
                break

            uci = move_to_uci(move)
            if board.white_to_move:
                print(f"{move_num}. {uci} (score: {score:.3f})", end='  ')
            else:
                print(f"{uci} (score: {score:.3f})")
                move_num += 1

            board.make_move(move)

def _print_legal_moves(board: Board, legal_moves: list):
    """Prints legal moves in a readable format."""
    uci_moves = [move_to_uci(m) for m in legal_moves]
    uci_moves.sort()
    print(f"Legal moves: {', '.join(uci_moves)}")

def _parse_uci(uci: str, legal_moves: list) -> int:
    """
    Parses a UCI string (e.g. 'e2e4', 'e7e8q') and returns the
    matching legal move integer, or None if not found.
    """
    if len(uci) < 4:
        return None

    file_map = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}

    try:
        from_file = file_map[uci[0]]
        from_rank = int(uci[1]) - 1
        to_file   = file_map[uci[2]]
        to_rank   = int(uci[3]) - 1
    except (KeyError, ValueError):
        return None

    from_sq = from_rank * 8 + from_file
    to_sq   = to_rank   * 8 + to_file

    # Handle promotion suffix
    promo_map = {
        'q': (FLAG_PROMOTE_QUEEN,  FLAG_PROMOTE_QUEEN_CAPTURE),
        'r': (FLAG_PROMOTE_ROOK,   FLAG_PROMOTE_ROOK_CAPTURE),
        'b': (FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_BISHOP_CAPTURE),
        'n': (FLAG_PROMOTE_KNIGHT, FLAG_PROMOTE_KNIGHT_CAPTURE),
    }
    promo_flags = promo_map.get(uci[4]) if len(uci) == 5 else None

    for move in legal_moves:
        if get_from_square(move) != from_sq: continue
        if get_to_square(move)   != to_sq:   continue
        flag = get_flag(move)

        if promo_flags:
            if flag in promo_flags:
                return move
        else:
            # For non-promotions, accept any matching from/to
            if flag not in (
                FLAG_PROMOTE_QUEEN,  FLAG_PROMOTE_ROOK,
                FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_KNIGHT,
                FLAG_PROMOTE_QUEEN_CAPTURE, FLAG_PROMOTE_ROOK_CAPTURE,
                FLAG_PROMOTE_BISHOP_CAPTURE, FLAG_PROMOTE_KNIGHT_CAPTURE,
            ):
                return move

    return None

# ====================================================
# Entry point
# ====================================================
if __name__ == "__main__":
    print("1. Self-play")
    print("2. Play against engine")
    choice = input("Choose mode: ").strip()

    if choice == '1':
        self_play()
    elif choice == '2':
        play_engine()