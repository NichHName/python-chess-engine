"""
    Program: pgn_parser.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpsose: Translate pgn files into readable moves for the engine.
"""

import re
import numpy as np
from core.bitboard import Board
from core.move_gen import get_all_legal_moves
from core.constants import (
    WHITE_SHORT_RIGHT, WHITE_LONG_RIGHT, BLACK_SHORT_RIGHT, BLACK_LONG_RIGHT, FLAG_SHORT_CASTLE,
    FLAG_LONG_CASTLE, FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_KNIGHT, FLAG_PROMOTE_QUEEN, FLAG_PROMOTE_ROOK,
    get_flag, get_from_square, get_to_square
    )

def board_to_features(board: Board) -> np.ndarray:
    """
    Press a Board into a feature vector:

    12 bitboards (one per piece type) flattened to 12*64 = 768 features\n
    + side to move (1)\n
    + castling rights (4)\n
    + en passant file (8, one-hot)\n
    = 781 features total
    """

    features = np.zeros(781, dtype=np.uint8)

    bitboards = [
        board.white_pawns,   board.white_knights, board.white_bishops,
        board.white_rooks,   board.white_queens,  board.white_king,
        board.black_pawns,   board.black_knights, board.black_bishops,
        board.black_rooks,   board.black_queens,  board.black_king,
    ]

    squares = np.arange(64, dtype=np.uint8)
    for i, bb in enumerate(bitboards):
        features[i*64:(i+1)*64] = ((bb >> squares) & 1).astype(np.uint8)
    
    features[768] = 1 if board.white_to_move else 0
    features[769] = 1 if board.castling_rights & WHITE_SHORT_RIGHT else 0
    features[770] = 1 if board.castling_rights & WHITE_LONG_RIGHT  else 0
    features[771] = 1 if board.castling_rights & BLACK_SHORT_RIGHT else 0
    features[772] = 1 if board.castling_rights & BLACK_LONG_RIGHT  else 0

    if board.en_passant_square is not None:
        features[773 + board.en_passant_square % 8] = 1

    return features.reshape(-1, 1)  # shape (781, 1) for network input

def load_pgn_training_data(pgn_file: str, max_positions: int = 100000) -> list:
    training_data = []
    games         = _parse_pgn_file(pgn_file)

    for game in games:
        if len(training_data) >= max_positions:
            break

        board = Board()
        board.set_starting_position()

        for san_move in game['moves']:
            if len(training_data) >= max_positions:
                break

            features = board_to_features(board)
            move     = _san_to_move(san_move, board)

            if move is None:
                break

            policy_index = _move_to_policy_index(move)
            training_data.append((features, policy_index))
            board.make_move(move)

    return training_data

# =========================================================
# Private helpers
# =========================================================
def _parse_pgn_file(pgn_file: str) -> list:
    """
    Splits a PGN file into a list of game dicts.
    Each dict has 'headers' and 'moves' (list of SAN strings).
    """
    games        = []
    current_game = {'headers': {}, 'raw_moves': []}
    move_lines   = []
    in_moves     = False

    with open(pgn_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('['):
                # New header
                if in_moves:
                    current_game['raw_moves'] = move_lines
                    games.append(current_game)
                    current_game = {'headers': {}, 'raw_moves': []}
                    move_lines   = []
                    in_moves     = False
                match = re.match(r'\[(.*?)\s+"(.*?)"\]', line)
                if match:
                    current_game['headers'][match.group(1)] = match.group(2)

            elif line:
                in_moves = True
                move_lines.append(line)
            
    # Handle last game if file doesn't end with empty line
    if move_lines:
        current_game['raw_moves'] = move_lines
        games.append(current_game)

    # Parse raw move text into clean SAN list for each game
    for game in games:
        raw = ' '.join(game['raw_moves'])
        game['moves'] = _parse_move_text(raw)

    return games

# Predefine so _parse_move_text doesn't have to every time
_RE_COMMENT_CURLY  = re.compile(r'\{[^}]*\}')
_RE_COMMENT_PAREN  = re.compile(r'\([^)]*\)')
_RE_ANNOTATION     = re.compile(r'\$\d+')
_RE_MOVE_NUMBERS   = re.compile(r'\d+\.+')
_RE_RESULT         = re.compile(r'(1-0|0-1|1/2-1/2|\*)')

def _parse_move_text(raw: str) -> list:
    """
    Converts raw PGN move text into a clean list of SAN strings.
    Removes move numbers, comments, annotations, and result strings.

    '1. e4 e5 2. Nf3 {comment} Nc6 3. Bb5 1-0' -> ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5']
    """
    raw = _RE_COMMENT_CURLY.sub('', raw)
    raw = _RE_COMMENT_PAREN.sub('', raw)
    raw = _RE_ANNOTATION.sub('', raw)
    raw = _RE_MOVE_NUMBERS.sub('', raw)
    raw = _RE_RESULT.sub('', raw)
    return raw.split()

def _move_to_policy_index(move: int) -> int:
    """
    Returns the flat policy index for a move (0-4095).
    Defers vector construction to the training batch builder.
    """
    return get_from_square(move) * 64 + get_to_square(move)

def _san_to_move(san: str, board: Board) -> int:
    
    if len(san) < 2:
        return None

    legal_moves = get_all_legal_moves(board)

    # Castling
    if san in ('O-O', '0-0', 'O-O+', '0-0+'):
        for move in legal_moves:
            if get_flag(move) == FLAG_SHORT_CASTLE:
                return move
        return None
    if san in ('O-O-O', '0-0-0', 'O-O-O+', '0-0-0+'):
        for move in legal_moves:
            if get_flag(move) == FLAG_LONG_CASTLE:
                return move
        return None

    san = san.rstrip('+#')

    # Promotion
    promotion_piece = None
    if '=' in san:
        promotion_piece = san[-1]
        san = san[:-2]

    # Piece type
    piece_map = {
        'N': 'knights', 'B': 'bishops',
        'R': 'rooks',   'Q': 'queens', 'K': 'king'
    }
    if san[0].isupper():
        piece_type = piece_map[san[0]]
        san        = san[1:]
    else:
        piece_type = 'pawns'

    san = san.replace('x', '')

    # Destination square
    to_sq = (int(san[-1]) - 1) * 8 + (ord(san[-2]) - ord('a'))

    # Disambiguation
    dis      = san[:-2]
    dis_file = ord(dis[0]) - ord('a') if dis and dis[0].islower()  else None
    dis_rank = int(dis[0]) - 1        if dis and dis[0].isdigit()  else None

    # Piece bitboard — precomputed once
    color    = 'white' if board.white_to_move else 'black'
    piece_bb = getattr(board, f"{color}_{piece_type}")

    promotion_flag_map = {
        'Q': FLAG_PROMOTE_QUEEN,  'R': FLAG_PROMOTE_ROOK,
        'B': FLAG_PROMOTE_BISHOP, 'N': FLAG_PROMOTE_KNIGHT,
    }

    for move in legal_moves:
        from_sq = get_from_square(move)

        # Ordered from cheapest to most expensive check
        if get_to_square(move)       != to_sq:            continue
        if not (piece_bb & (1 << from_sq)):                continue
        if dis_file is not None and from_sq % 8 != dis_file: continue
        if dis_rank is not None and from_sq // 8 != dis_rank: continue
        if promotion_piece and get_flag(move) != promotion_flag_map.get(promotion_piece): continue

        return move

    return None