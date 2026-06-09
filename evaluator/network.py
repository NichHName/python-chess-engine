"""
    Program: network.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: This program stores the architecture of the NN to train.
             Uses NumPy as the only difference is efficiency; all structure
             is still built by hand.
"""

import numpy as np

# =========================================================
# Activation functions
# =========================================================
def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation function for an array (layer).
    """
    return np.maximum(0, x)

def relu_backward(x: np.ndarray):
    """
    Gradient of ReLU activation function for an array (layer).
    """
    return (x > 0).astype(float)

def tanh_backward(x: np.ndarray) -> np.ndarray:
    """
    Gradient of tanh activation function for an array (layer).
    """
    return 1 - (np.tanh(x))**2 # More numerically stable than exponential form

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function for an array (layer).
    """
    shifted = x - np.max(x)  # subtract max for stability
    ez      = np.exp(shifted)
    return ez / np.sum(ez)

# =========================================================
# Weight initialization
# =========================================================
def init_weights_he(in_size: int, out_size: int) -> tuple:
    """
    Initialize a layer's weights using He initialization: for non-tanh layers.
    """
    # He initialization for ReLU layers. Prevents vanishing/exploding gradients
    scale = np.sqrt(2.0 / in_size)
    W     = np.random.randn(out_size, in_size) * scale
    b     = np.zeros((out_size, 1))
    return W, b

def init_weights_xavier(in_size: int, out_size: int) -> tuple:
    """
    Initialize a layer's weights using Xavier initialization: for tanh layers.
    """
    # Xavier initialization for tanh. Slightly simpler size tanh doesn't zero out many activations
    scale = np.sqrt(1.0 / in_size)
    W     = np.random.randn(out_size, in_size) * scale
    b     = np.zeros((out_size, 1))
    return W, b

# =========================================================
# Layer class
# =========================================================
class DenseLayer:
    def __init__(self, in_size: int, out_size: int, activation: str = 'relu'):
        """
        Initialize a Dense Layer of the network with a specified activation function.
        activation:\n 
                    'relu'    - Hidden layers \n
                    'tanh'    - Value head    \n
                    'softmax' - Policy head   \n
                    'linear'  - No activation
        """
        self.activation = activation

        if activation != 'tanh':
            self.W, self.b = init_weights_he(in_size, out_size)
        else:
            self.W, self.b = init_weights_xavier(in_size, out_size)
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.x = None
        self.z = None
        self.a = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer. stores x and z for backward pass.
        """
        self.x = x
        self.z = self.W @ x + self.b

        if self.activation == 'relu':
            self.a = relu(self.z)
        elif self.activation == 'tanh':
            self.a = np.tanh(self.z)
        elif self.activation == 'softmax':
            self.a = softmax(self.z.flatten()).reshape(-1, 1)
        elif self.activation == 'linear':
            self.a = self.z
        
        return self.a
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through this layer. Computes gradients for W and b,
        returns gradient for previous layer.

        grad_z = grad_output * activation'(z)   element-wise \n
        grad_W = grad_z @ x^T                                \n
        grad_b = grad_z                                      \n
        grad_x = W^T @ grad_z                   passed to previous layer
        """

        if self.activation == 'relu':
            grad_z = grad_output * relu_backward(self.z)
        elif self.activation == 'tanh':
            grad_z = grad_output * tanh_backward(self.z)
        elif self.activation == 'softmax':
            # Softmax backward is handled in the loss function
            # (cross-entropy + softmax combine cleanly)
            grad_z = grad_output
        elif self.activation == 'linear':
            grad_z = grad_output

        self.dW = grad_z @ self.x.T
        self.db = grad_z

        grad_x = self.W.T @ grad_z

        return grad_x
    
    def zero_grad(self):
        """
        Zeros out gradient for the next loop: called at beginning of a training loop.
        """
        self.dW = np.zeros_like(self.dW)
        self.db = np.zeros_like(self.db)

# =========================================================
# Network class
# =========================================================
class ChessNetwork:
    def __init__(self,
                 in_size:           int = 781,
                 trunk_sizes:       list = [512, 256, 128],
                 value_head_sizes:  list = [64],
                 policy_head_sizes: list = [64],
                 policy_out:        int  = 4096):
        """
        Architecture:
            input (781)
                - shared trunk (dense + ReLU layers)
                    - value head  -> tanh -> scalar (-1 to 1)
                    - policy head -> softmax -> move probabilities (4096)

        in_size:        board feature vector size (781) \n
        trunk_sizes:       hidden layer sizes for shared trunk \n
        value_head_sizes:  hidden layer sizes for value head \n
        policy_head_sizes: hidden layer sizes for policy head \n
        policy_out:     number of possible moves (64*64 = 4096)

        Modeled after AlphaZero training scheme.
        """

        self.trunk = []
        prev_size = in_size

        for size in trunk_sizes:
            self.trunk.append(DenseLayer(prev_size, size, activation='relu'))
            prev_size = size

        trunk_output_size = prev_size

        self.value_head = []
        # prev_size = trunk_output_size
        for size in value_head_sizes:
            self.value_head.append(DenseLayer(prev_size, size, activation='relu'))
            prev_size = size

        self.value_head.append(DenseLayer(prev_size, 1, activation='tanh'))

        self.policy_head = []
        prev_size = trunk_output_size
        for size in policy_head_sizes:
            self.policy_head.append(DenseLayer(prev_size, size, activation='relu'))
            prev_size = size
        
        self.policy_head.append(DenseLayer(prev_size, policy_out, activation='softmax'))

        self.in_size =           in_size
        self.trunk_sizes =       trunk_sizes
        self.value_head_sizes =  value_head_sizes
        self.policy_head_sizes = policy_head_sizes
        self.policy_out =        policy_out

    def forward(self, x: np.ndarray) -> tuple:
        """
        Full forward pass through the network.

        x: board feature vector, shape (781, 1)
        returns: (value, policy)
            value:  scalar in (-1, 1)
            policy: probability distribution over 4096 moves
        """
        # Shared trunk
        trunk_out = x
        for layer in self.trunk:
            trunk_out = layer.forward(trunk_out)

        # Value head
        value_out = trunk_out
        for layer in self.value_head:
            value_out = layer.forward(value_out)

        # Policy head
        policy_out = trunk_out
        for layer in self.policy_head:
            policy_out = layer.forward(policy_out)

        return value_out, policy_out

    def backward(self, value_grad: np.ndarray, policy_grad: np.ndarray):
        """
        Full backward pass through the network.
        Propagates gradients from both heads back through the shared trunk.

        grad_value:  gradient from value loss, shape (1, 1)
        grad_policy: gradient from policy loss, shape (4096, 1)
        """

        # Backprop through value head
        grad = value_grad
        for layer in reversed(self.value_head):
            grad = layer.backward(grad)
        grad_trunk_from_value = grad

        # Backprop through policy head
        grad = policy_grad
        for layer in reversed(self.policy_head):
            grad = layer.backward(grad)
        grad_trunk_from_policy = grad

        # Add because both heads feed from the same forward pass
        grad_trunk = grad_trunk_from_value + grad_trunk_from_policy

        for layer in reversed(self.trunk):
            grad_trunk = layer.backward(grad_trunk)


    def zero_grad(self):
        """
        Clears all accumulated gradients across all layers
        """
        for layer in self.trunk:
            layer.zero_grad()
        for layer in self.value_head:
            layer.zero_grad()
        for layer in self.policy_head:
            layer.zero_grad()

    def get_all_layers(self) -> list:
        """
        Returns all layers in order (for optimizers).
        """
        return self.trunk + self.value_head + self.policy_head

    def save(self, path: str):
        """
        Serializes all weights and biases to disk using NumPy.
        Also saves architecture config so load() can reconstruct.
        """
        data = {
            'config': {
                'in_size':           self.in_size,
                'trunk_sizes':       self.trunk_sizes,
                'value_head_sizes':  self.value_head_sizes,
                'policy_head_sizes': self.policy_head_sizes,
                'policy_out':        self.policy_out,
            }
        }

        for i, layer in enumerate(self.trunk):
            data[f'trunk_{i}_W'] = layer.W
            data[f'trunk_{i}_b'] = layer.b

        for i, layer in enumerate(self.value_head):
            data[f'value_{i}_W'] = layer.W
            data[f'value_{i}_b'] = layer.b

        for i, layer in enumerate(self.policy_head):
            data[f'policy_{i}_W'] = layer.W
            data[f'policy_{i}_b'] = layer.b

        np.savez(path, **data)
        print(f"Network saved to {path}.npz")

    @classmethod
    def load(cls, path: str):
        """
        Reconstructs a ChessNetwork from a saved .npz file.
        """
        data   = np.load(path, allow_pickle=True)
        config = data['config'].item()

        network = cls(
            in_size        = config['in_size'],
            trunk_sizes       = config['trunk_sizes'],
            value_head_sizes  = config['value_head_sizes'],
            policy_head_sizes = config['policy_head_sizes'],
            policy_out     = config['policy_out'],
        )

        for i, layer in enumerate(network.trunk):
            layer.W = data[f'trunk_{i}_W']
            layer.b = data[f'trunk_{i}_b']

        for i, layer in enumerate(network.value_head):
            layer.W = data[f'value_{i}_W']
            layer.b = data[f'value_{i}_b']

        for i, layer in enumerate(network.policy_head):
            layer.W = data[f'policy_{i}_W']
            layer.b = data[f'policy_{i}_b']

        print(f"Network loaded from {path}")
        return network