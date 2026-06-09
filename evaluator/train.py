"""
    Program: train.py
    Date: May 2026
    Author: Nicholas Harsell
    Purpose: Training pipeline for the chess network.
             Phase 1: Supervised learning from GM games.
             Phase 2: Self-play reinforcement learning (future).
"""

import numpy as np
import os
from data.pgn_parser import load_pgn_training_data, board_to_features
from evaluator.network import ChessNetwork
from evaluator.loss_optim import combined_loss, SGD

# ====================================================
# Configuration
# ====================================================
BATCH_SIZE     = 64
EPOCHS         = 10
LEARNING_RATE  = 0.01
TRAIN_SPLIT    = 0.8
SAVE_PATH      = 'models/network'

# ====================================================
# Data preparation
# ====================================================
def prepare_batches(training_data: list, batch_size: int) -> list:
    """
    Converts flat training data into shuffled minibatches.
    Each batch is a list of (features, policy_index) tuples.
    """
    np.random.shuffle(training_data)
    batches = []
    for i in range(0, len(training_data), batch_size):
        batches.append(training_data[i:i + batch_size])
    return batches

def split_data(training_data: list, train_split: float) -> tuple:
    """
    Splits data into training and validation sets.
    Returns (train_data, val_data)
    """
    split_idx  = int(len(training_data) * train_split)
    train_data = training_data[:split_idx]
    val_data   = training_data[split_idx:]
    return train_data, val_data

def build_policy_vector(policy_index: int) -> np.ndarray:
    """
    Converts a policy index to a one-hot vector of shape (4096, 1).
    Called during batch construction rather than data loading.
    """
    policy = np.zeros((4096, 1), dtype=np.float32)
    policy[policy_index] = 1.0
    return policy

# ====================================================
# Evaluation on validation set
# ====================================================
def evaluate_validation(network: ChessNetwork, val_data: list) -> tuple:
    """
    Computes average value and policy loss on the validation set.
    Returns (avg_value_loss, avg_policy_loss, avg_total_loss)
    
    During supervised pretraining we have no value targets
    (we don't know who won from mid-game positions), so value
    loss is skipped here. Only policy loss is measured.
    """
    total_policy_loss = 0.0
    total_value_loss  = 0.0

    for features, policy_index in val_data:
        value_pred, policy_pred = network.forward(features)

        # Policy loss — did the network predict the GM move?
        policy_target = build_policy_vector(policy_index)
        p_loss = _cross_entropy(policy_pred, policy_target)
        total_policy_loss += p_loss

        # Value target is 0 (unknown) during supervised training
        v_loss = float((value_pred - 0.0) ** 2)
        total_value_loss += v_loss

    n              = len(val_data)
    avg_policy     = total_policy_loss / n
    avg_value      = total_value_loss  / n
    avg_total      = avg_policy + avg_value

    return avg_value, avg_policy, avg_total

def _cross_entropy(predicted: np.ndarray, target: np.ndarray) -> float:
    """Local cross entropy for validation — no gradient needed."""
    clipped = np.clip(predicted, 1e-10, 1.0)
    return float(-np.sum(target * np.log(clipped)))

# ====================================================
# Training step
# ====================================================
def train_step(network:   ChessNetwork,
               optimizer: SGD,
               batch:     list) -> float:
    """
    Runs one gradient update on a single minibatch.
    Returns the average loss for this batch.
    """
    total_loss = 0.0
    network.zero_grad()

    for features, policy_index in batch:
        # Forward pass
        value_pred, policy_pred = network.forward(features)

        # Build targets
        policy_target = build_policy_vector(policy_index)
        value_target  = 0.0  # unknown during supervised training

        # Compute loss and gradients
        loss, grad_value, grad_policy = combined_loss(
            value_pred,   value_target,
            policy_pred,  policy_target,
            value_weight  = 0.1,   # downweight value during supervised phase
            policy_weight = 1.0,
        )

        total_loss += loss

        # Backward pass — accumulates gradients
        network.backward(
            grad_value.reshape(-1, 1),
            grad_policy.reshape(-1, 1)
        )

    # Average gradients over batch
    for layer in network.get_all_layers():
        layer.dW /= len(batch)
        layer.db /= len(batch)

    # Update weights
    optimizer.update(network)

    return total_loss / len(batch)

# ====================================================
# Main supervised training loop
# ====================================================
def train_supervised(pgn_file:      str,
                     max_positions: int   = 100000,
                     epochs:        int   = EPOCHS,
                     batch_size:    int   = BATCH_SIZE,
                     learning_rate: float = LEARNING_RATE,
                     save_path:     str   = SAVE_PATH) -> ChessNetwork:
    """
    Full supervised training pipeline.
    
    1. Load and parse PGN file\n
    2. Split into train/validation\n
    3. Train for N epochs\n
    4. Print loss at each epoch\n
    5. Save best network
    """
    print("=" * 50)
    print("Supervised Training Pipeline")
    print("=" * 50)

    # Load data
    print(f"\nLoading PGN data from {pgn_file}...")
    training_data = load_pgn_training_data(pgn_file, max_positions)
    print(f"Loaded {len(training_data)} positions")

    if not training_data:
        print("No training data found. Check PGN file path.")
        return None

    # Split data
    train_data, val_data = split_data(training_data, TRAIN_SPLIT)
    print(f"Train: {len(train_data)} positions")
    print(f"Val:   {len(val_data)} positions")

    # Initialize network and optimizer
    network   = ChessNetwork()
    optimizer = SGD(learning_rate=learning_rate)

    best_val_loss = float('inf')
    best_epoch    = 0

    print(f"\nTraining for {epochs} epochs, batch size {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 50)

    # Training loop
    for epoch in range(1, epochs + 1):
        # Shuffle and batch training data
        batches    = prepare_batches(train_data, batch_size)
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(batches):
            loss        = train_step(network, optimizer, batch)
            epoch_loss += loss

            # Progress indicator every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(batches)} "
                      f"| Loss: {loss:.4f}")

        avg_train_loss = epoch_loss / len(batches)

        # Validation
        val_value_loss, val_policy_loss, val_total_loss = \
            evaluate_validation(network, val_data)

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train loss:        {avg_train_loss:.4f}")
        print(f"  Val policy loss:   {val_policy_loss:.4f}")
        print(f"  Val value loss:    {val_value_loss:.4f}")
        print(f"  Val total loss:    {val_total_loss:.4f}")

        # Save best network
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_epoch    = epoch
            os.makedirs('models', exist_ok=True)
            network.save(save_path)
            print(f"  New best model saved (val loss: {best_val_loss:.4f})")

        print("-" * 50)

    print(f"\nTraining complete.")
    print(f"Best model at epoch {best_epoch} with val loss {best_val_loss:.4f}")
    print(f"Saved to {save_path}.npz")

    return network

# ====================================================
# Network integration with engine
# ====================================================
def load_network_for_engine(path: str) -> ChessNetwork:
    """
    Loads a trained network for use in the engine.
    Returns None if no saved network exists,
    allowing the engine to fall back to base_layers.
    """
    npz_path = path + '.npz'
    if not os.path.exists(npz_path):
        print(f"No saved network found at {npz_path}. Using base evaluator.")
        return None
    return ChessNetwork.load(npz_path)

def network_evaluate(network: ChessNetwork, board) -> float:
    """
    Evaluates a board position using the trained network.
    Returns a score in centipawns from white's perspective,
    scaled from the network's (-1, 1) output to match
    the hand-coded evaluator's range.
    
    Falls back to None if network is unavailable,
    letting the caller use base_layers instead.
    """
    if network is None:
        return None

    features              = board_to_features(board)
    value_pred, _         = network.forward(features)
    scalar                = float(value_pred.flatten()[0])

    # Scale from (-1, 1) to centipawns
    # A queen is worth 900cp, so full win ≈ queen advantage
    return scalar * 900

# Entry point for standalone training
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train.py <pgn_file> [max_positions]")
        sys.exit(1)

    pgn_file      = sys.argv[1]
    max_positions = int(sys.argv[2]) if len(sys.argv) > 2 else 100000

    network = train_supervised(pgn_file, max_positions)