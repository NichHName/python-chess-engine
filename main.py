import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from tests.perft import verify_perft
from core.bitboard import Board

board = Board()
board.set_starting_position()
verify_perft(4)