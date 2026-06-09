"""
    Program: loss_optim.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: This program stores the actual functions for gradient descent, MSE
             and cross-entropy loss, and other estimations.
"""

import numpy as np
from evaluator.network import ChessNetwork

# ====================================================
# Loss functions
# ====================================================
def mse_loss(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Mean Squared Error loss for the value head.
    
    predicted: network value output, shape (1, 1), range (-1, 1)\n
    target:    actual game outcome, scalar: 1=white wins, 0=draw, -1=black wins
    
    L = (predicted - target)^2
    """
    return float((predicted - target)**2)

def mse_loss_backward(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE loss with respect to predicted.
    dL/dp = 2 * (p - target)
    """
    return 2.0 * (predicted - target)

def cross_entropy_loss(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Cross entropy loss for the policy head.

    predicted: softmax output, shape (4096, 1) — move probabilities\n
    target:    target distribution, shape (4096, 1)
               from supervised learning: one-hot vector (GM move = 1)
               from self-play: MCTS visit count distribution

    L = -sum(target * log(predicted))

    log is clipped to prevent log(0) = -inf
    """
    predicted_clipped = np.clip(predicted, 1e-10, 1.0)
    return float(-np.sum(target * np.log(predicted_clipped)))

def cross_entropy_loss_backward(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Gradient of cross entropy loss combined with softmax.
    
    The combined softmax + cross entropy gradient simplifies cleanly to:
    dL/dz = predicted - target
    
    This is the key simplification — instead of computing the full
    softmax Jacobian (expensive), the combined gradient is trivial.
    This is why softmax backward in DenseLayer passes grad straight through.
    """
    return predicted - target

def combined_loss(value_pred:    np.ndarray,
                  value_target:  float,
                  policy_pred:   np.ndarray,
                  policy_target: np.ndarray,
                  value_weight:  float = 1.0,
                  policy_weight: float = 1.0) -> tuple:
    """
    Combined value and policy loss.\n
    AlphaZero formula: L = value_loss + policy_loss

    Returns (total_loss, grad_value, grad_policy) so the caller
    has everything needed to call network.backward() directly.
    """
    v_loss   = mse_loss(value_pred, value_target)
    p_loss   = cross_entropy_loss(policy_pred, policy_target)
    total    = value_weight * v_loss + policy_weight * p_loss

    grad_value  = mse_loss_backward(value_pred, value_target) * value_weight
    grad_policy = cross_entropy_loss_backward(policy_pred, policy_target) * policy_weight

    return total, grad_value, grad_policy

# ====================================================
# Optimizers
# ====================================================
class SGD:
    """
    Vanilla stochastic gradient descent.
    W = W - lr * dW
    
    Simple but slow to converge, useful as a baseline.
    """
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def update(self, network: ChessNetwork):
        for layer in network.get_all_layers():
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db

class SGDMomentum:
    """
    SGD with momentum.
    Maintains a velocity term that accumulates gradients over time,
    smoothing updates and helping escape shallow local minima.

    v = momentum * v - lr * dW \n
    W = W + v
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr       = learning_rate
        self.momentum = momentum
        self.velocity = {}  # stores velocity for each layer

    def update(self, network: ChessNetwork):
        for i, layer in enumerate(network.get_all_layers()):
            if i not in self.velocity:
                # Initialize velocity to zero on first update
                self.velocity[i] = {
                    'W': np.zeros_like(layer.W),
                    'b': np.zeros_like(layer.b)
                }

            self.velocity[i]['W'] = (self.momentum * self.velocity[i]['W']
                                     - self.lr * layer.dW)
            self.velocity[i]['b'] = (self.momentum * self.velocity[i]['b']
                                     - self.lr * layer.db)

            layer.W += self.velocity[i]['W']
            layer.b += self.velocity[i]['b']

class Adam:
    """
    Adaptive Moment Estimation.
    Maintains per-parameter learning rates by tracking:\n
        m: first moment  (mean of gradients)\n
        v: second moment (mean of squared gradients)

    m = beta1 * m + (1 - beta1) * dW\n
    v = beta2 * v + (1 - beta2) * dW^2
    
    Bias correction (important in early training when m and v are near zero):\n
    m_hat = m / (1 - beta1^t)\n
    v_hat = v / (1 - beta2^t)
    
    Weight update:\n
    W = W - lr * m_hat / (sqrt(v_hat) + epsilon)

    Standard hyperparameters from the original Adam paper:\n
    lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
    """
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1:         float = 0.9,
                 beta2:         float = 0.999,
                 epsilon:       float = 1e-8):
        self.lr      = learning_rate
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self.t       = 0       # timestep for bias correction
        self.m       = {}      # first moments
        self.v       = {}      # second moments

    def update(self, network: ChessNetwork):
        self.t += 1

        for i, layer in enumerate(network.get_all_layers()):
            if i not in self.m:
                # Initialize moments to zero on first update
                self.m[i] = {
                    'W': np.zeros_like(layer.W),
                    'b': np.zeros_like(layer.b)
                }
                self.v[i] = {
                    'W': np.zeros_like(layer.W),
                    'b': np.zeros_like(layer.b)
                }

            # Update first moment (mean)
            self.m[i]['W'] = self.beta1 * self.m[i]['W'] + (1 - self.beta1) * layer.dW
            self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1 - self.beta1) * layer.db

            # Update second moment (variance)
            self.v[i]['W'] = self.beta2 * self.v[i]['W'] + (1 - self.beta2) * layer.dW ** 2
            self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1 - self.beta2) * layer.db ** 2

            # Bias correction
            m_hat_W = self.m[i]['W'] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m[i]['b'] / (1 - self.beta1 ** self.t)
            v_hat_W = self.v[i]['W'] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v[i]['b'] / (1 - self.beta2 ** self.t)

            # Update weights
            layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)