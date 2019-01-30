from dataset import *
from preprocess import *
from gensim.models import Word2Vec
from setup import CONFIG
from models import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

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
    "1d": CNN1d(129, 2),
    "2d": CNN2d(32, 2),
    "lstm": RNN(vocab_size, model_10,
                CONFIG["rnn"]["m"], CONFIG["rnn"]["n"],
                2, 16, 2, mode="lstm"),
    "gru": RNN(vocab_size, model_10,
               CONFIG["rnn"]["m"], CONFIG["rnn"]["n"],
               2, 16, 2, mode="gru"),
    "conv1drnn": CONV1dRNN(vocab_size, model_10,
                           CONFIG["conv_1d_rnn"]["m"], CONFIG["conv_1d_rnn"]["n"],
                           2, 16, 2, mode="gru"),
    "conv2drnn": CONV2dRNN(vocab_size, model_44,
                           CONFIG["conv_2d_rnn"]["m"], CONFIG["conv_2d_rnn"]["n"],
                           2, CONFIG["conv_2d_rnn"]["m"], 2, mode="gru")
}

# dataset


def test_dataloader(dl, rna):
    ex = next(iter(dl))
    print(ex[0].shape)
    out = rna(ex[0])
    print(out.shape)


train_data, test_data, train_target, test_target = train_test_split(
    corpus, target, test_size=0.25)

rnn_train = Data(train_data, train_target, model_10,
                 max_sentence, 10, stopw, word2id, "rnn")
rnn_test = Data(test_data, test_target, model_10,
                max_sentence, 10, stopw, word2id, "rnn")

dl_rnn_train = DataLoader(rnn_train, batch_size=1, shuffle=True)
dl_rnn_test = DataLoader(rnn_test, batch_size=1, shuffle=False)

print("rnn train")
test_dataloader(dl_rnn_train, models["lstm"])
print("rnn test")
test_dataloader(dl_rnn_test, models["gru"])

print("conv1d rnn test")
test_dataloader(dl_rnn_train, models["conv1drnn"])
print("conv1d rnn test")
test_dataloader(dl_rnn_test, models["conv1drnn"])

print("conv2d rnn test")
test_dataloader(dl_rnn_train, models["conv2drnn"])
print("conv2d rnn test")
test_dataloader(dl_rnn_test, models["conv2drnn"])


conv1d_train = Data(train_data, train_target, model_10,
                    max_sentence, 10, stopw, word2id, "1d")
conv1d_test = Data(test_data, test_target, model_10,
                   max_sentence, 10, stopw, word2id, "1d")

dl_conv1d_train = DataLoader(conv1d_train,
                             batch_size=int(conv1d_train._len/3),
                             shuffle=True)
dl_conv1d_test = DataLoader(conv1d_test,
                            batch_size=conv1d_test._len,
                            shuffle=True)

print("conv1d train")
test_dataloader(dl_conv1d_train, models["1d"])
print("conv1d test")
test_dataloader(dl_conv1d_test, models["1d"])

conv2d_train = Data(train_data, train_target, model_44,
                    max_sentence, max_sentence, stopw, word2id, "2d")
conv2d_test = Data(test_data, test_target, model_44,
                   max_sentence, max_sentence, stopw, word2id, "2d")

dl_conv2d_train = DataLoader(conv2d_train,
                             batch_size=int(conv2d_train._len/3),
                             shuffle=True)
dl_conv2d_test = DataLoader(conv2d_test,
                            batch_size=conv2d_test._len,
                            shuffle=True)

print("conv2d train")
test_dataloader(dl_conv2d_train, models["2d"])
print("conv2d test")
test_dataloader(dl_conv2d_test, models["2d"])
