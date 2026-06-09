"""
    Program: eval_router.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: Routes evaluation calls to the network or base evaluator
             depending on whether a trained network is available.
"""

from evaluator.base_layers import evaluate as base_evaluate
from evaluator.train import load_network_for_engine, network_evaluate

class EvalRouter:
    def __init__(self, model_path: str = 'models/network'):
        self.network = load_network_for_engine(model_path)
        if self.network:
            print("Using trained network for evaluation.")
        else:
            print("Using hand-coded evaluator.")

    def evaluate(self, board) -> float:
        net_score = network_evaluate(self.network, board)
        return net_score if net_score is not None else base_evaluate(board)

# Global instance
_router = EvalRouter()

def evaluate(board) -> float:
    return _router.evaluate(board)