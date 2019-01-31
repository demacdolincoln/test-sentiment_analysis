from dataset import *
from preprocess import *
from gensim.models import Word2Vec
from setup import CONFIG
from models import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from train import train

corpus, target, stopw = cleantext("dataset/*labelled.txt")

model_10 = Word2Vec().wv.load_word2vec_format("word2vec/skipgram_10.vec")
model_44 = Word2Vec().wv.load_word2vec_format("word2vec/skipgram_44.vec")

id2word = model_10.index2word
word2id = {k: v for v, k in enumerate(id2word)}
vocab_size = len(id2word)

max_sentence = 0
for i in corpus:
    len_sentence = len(i)
    if len_sentence > max_sentence:
        max_sentence = len_sentence
print("max_sentence: ", max_sentence)

models = {
    "conv_1d": CNN1d(129, 2),
    "conv_2d": CNN2d(32, 2),
    "lstm": RNN(vocab_size, model_10,
                CONFIG["rnn"]["m"], CONFIG["rnn"]["n"],
                2, 16, 2, mode="lstm"),
    "gru": RNN(vocab_size, model_10,
               CONFIG["rnn"]["m"], CONFIG["rnn"]["n"],
               2, 16, 2, mode="gru"),
    "conv_1d_rnn": CONV1dRNN(vocab_size, model_10,
                             CONFIG["conv_1d_rnn"]["m"], CONFIG["conv_1d_rnn"]["n"],
                             2, 16, 2, mode="gru"),
    "conv_2d_rnn": CONV2dRNN(vocab_size, model_44,
                             CONFIG["conv_2d_rnn"]["m"], CONFIG["conv_2d_rnn"]["n"],
                             2, CONFIG["conv_2d_rnn"]["m"], 2, mode="gru")
}

# dataset

dloaders = {}  # data loaders

train_data, test_data, train_target, test_target = train_test_split(
    corpus, target, test_size=0.25)

# rnn
rnn_train = Data(train_data, train_target, model_10,
                 max_sentence, 10, stopw, word2id, "rnn")
rnn_test = Data(test_data, test_target, model_10,
                max_sentence, 10, stopw, word2id, "rnn")

dloaders["rnn_train"] = DataLoader(rnn_train,
                                   batch_size=1,
                                   shuffle=True)
dloaders["rnn_test"] = DataLoader(rnn_test,
                                  batch_size=1,
                                  shuffle=False)

# conv1d
conv1d_train = Data(train_data, train_target, model_10,
                    max_sentence, 10, stopw, word2id, "1d")
conv1d_test = Data(test_data, test_target, model_10,
                   max_sentence, 10, stopw, word2id, "1d")

dloaders["conv_1d_train"] = DataLoader(conv1d_train,
                                       batch_size=int(conv1d_train._len/3),
                                       shuffle=True)
dloaders["conv_1d_test"] = DataLoader(conv1d_test,
                                      batch_size=conv1d_test._len,
                                      shuffle=True)

# conv2d
conv2d_train = Data(train_data, train_target, model_44,
                    max_sentence, max_sentence, stopw, word2id, "2d")
conv2d_test = Data(test_data, test_target, model_44,
                   max_sentence, max_sentence, stopw, word2id, "2d")

dloaders["conv_2d_train"] = DataLoader(conv2d_train,
                                       batch_size=int(conv2d_train._len/3),
                                       shuffle=True)
dloaders["conv_2d_test"] = DataLoader(conv2d_test,
                                      batch_size=conv2d_test._len,
                                      shuffle=True)

for nn_name in CONFIG.keys():
    vec_model = None
    if CONFIG[nn_name]["n"] == 10:
        vec_model = model_10
    else:
        vec_model = model_44

    if nn_name == "rnn":
        for rnn_type in ["gru", "lstm"]:
            train(models[rnn_type], vec_model,
                  dloaders["rnn_train"], dloaders["rnn_test"],
                  CONFIG[nn_name], f"rnn_{rnn_type}")
    elif "rnn" in nn_name:
        train(models[nn_name], vec_model,
              dloaders["rnn_train"], dloaders["rnn_test"],
              CONFIG[nn_name], nn_name)
    else:
        train(models[nn_name], vec_model,
              dloaders[f"{nn_name}_train"], dloaders[f"{nn_name}_test"],
              CONFIG[nn_name], nn_name)
