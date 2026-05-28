# Pure-Python Chess Engine

The goal of this project is a pure Python implementation of a Chess engine with as little help from outside packages as possible (and zero outside programming langauges). The game itself, move evaluation, neural network architecture, training, and main driver are all hand-coded. This project is a challenge for the author for several reasons. For one: being written entirely in Python, extreme optimization is required for such a project to give any meaningful results in a reasonable amount of time. Simple AI-generated code will not do the trick. Therefore, this project requires knowledge of
1) Tensor/matrix/vector calculus,
2) Data structures and their implementations for rigid optimization,
3) Neural network architecture,
4) Algorithms for efficiently finding and evaluating legal moves,
5) Game theory strategies for optimal move choice,
6) Data processing, general I/O,
and more. This is a work-in-progress!

Work will be released in chunks (mostly corresponding to the timeline below). At the end of the project, I will release a detailed research-paper-formatted document to explain design choices, particular challenges associated with such a project, and advice for building your own chess engine.

## Current Workflow

### Phase 1: The Game (Complete)
1) constants -> bitboard -> magic -> move_gen -> zobrist
2) Benchmark move_gen w/ depth 3, 4, 5 (using `perft.py` to ensure all moves are correctly identified)
3) Complete zobrist hashing

### Phase 2: Searching (Complete)
4) minimax using move_gen and a temporary naive piece-value evaluator function
5) Pruning logic

### Phase 3: Learning (In-Progress)
6) base_layers -> loss_optim (Linear layers, ReLU, MSE/proprietary loss?)
7) network -> train
8) Train on sine-wave fitting to test gradient loss convergence

### Phase 4: Training (Incomplete)
9) pgn_parser (gather data, vectorization)
10) Train on new data

### Phase 5: Implementation (Incomplete)
11) Replace material counter in minimax with forward pass of trained NN
12) Add transposition tables