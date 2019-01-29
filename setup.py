CONFIG = {
    "2d": {
        "m": 44,
        "n": 44,
        "vec_path": f"word2vec/skipgram_44.vec",
        "epochs": 100,
        "lr": 1e-3,
        "save_path": "saved_models/conv2d.pth"
    },
    "1d": {
        "m": 44,
        "n": 10,
        "vec_path": f"word2vec/skipgram_10.vec",
        "epochs": 100,
        "lr": 1e-3,
        "save_path": "saved_models/conv1d.pth"
    },
    "rnn": {
        "m": 44,
        "n": 10,
        "vec_path": f"word2vec/skipgram_10.vec",
        "epochs": 100,
        "lr": 1e-4,
        "save_path": "saved_models/rnn_{_type}.pth"
    },
    "conv_1d_rnn": {
        "m": 44,
        "n": 10,
        "vec_path": f"word2vec/skipgram_10.vec",
        "epochs": 100,
        "lr": 1e-5,
        "save_path": "saved_models/conv1d_rnn.pth"
    },
    "conv_2d_rnn": {
        "m": 44,
        "n": 44,
        "vec_path": f"word2vec/skipgram_44.vec",
        "epochs": 100,
        "lr": 1e-5,
        "save_path": "saved_models/conv1d_rnn.pth"
    }
}


if __name__ == "__main__":

    from preprocess import cleantext, save_word2vec

    ex_1d = CONFIG["1d"]
    ex_2d = CONFIG["2d"]

    corpus, target, stopw = cleantext("dataset/*labelled.txt")
    for i in [ex_1d, ex_2d]:
        save_word2vec(corpus, i["n"], 1,
                      2000, i["vec_path"])
        print(i["vec_path"], ": ok")
