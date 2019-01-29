CONFIG = {
    "2d": {
        "m": max_sentence,
        "n": max_sentence,
        "path": f"word2vec/skipgram_{max_sentence}.vec",
        "epochs": 100,
        "lr": 1e-3,
        "save_path": "saved_models/conv2d.pth"
    },
    "1d": {
        "m": max_sentence,
        "n": 10,
        "path": f"word2vec/skipgram_10.vec",
        "epochs": 100,
        "lr": 1e-3,
        "save_path": "saved_models/conv1d.pth"
    }
    "rnn": {
        "m": max_sentence,
        "n": 10,
        "path": f"word2vec/skipgram_10.vec",
        "epochs": 100,
        "lr": 1e-4,
        "save_path": "saved_models/rnn_{_type}.pth"
    },
    "conv_1d_rnn": {
        "m": max_sentence,
        "n": 10,
        "path": f"word2vec/skipgram_10.vec",
        "epochs": 100,
        "lr": 1e-5,
        "save_path": "saved_models/conv1d_rnn.pth"
    },
    "conv_2d_rnn": {
        "m": max_sentence,
        "n": max_sentence,
        "path": f"word2vec/skipgram_10.vec",
        "epochs": 100,
        "lr": 1e-5,
        "save_path": "saved_models/conv1d_rnn.pth"
    }
}

