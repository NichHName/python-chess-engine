from evaluator.train import train_supervised

games = ['data/twic_1636_1646_69271.pgn']
for game in games:
    network = train_supervised(
        pgn_file      = 'data/twic_1636_1646_69271.pgn',
        max_positions = 500000,
        epochs        = 20,
        learning_rate = 0.001,
        batch_size    = 64,
        save_path     = 'models/network'
    )