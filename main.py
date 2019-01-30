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
train_data, test_data, train_target, test_target = train_test_split(
    corpus, target, test_size=0.25)

rnn_train = Data(train_data, train_target, model_10,
                 max_sentence, 10, stopw, word2id, "rnn")
rnn_test = Data(test_data, test_target, model_10,
                 max_sentence, 10, stopw, word2id, "rnn")
print(rnn_train[4][0].shape)
print(rnn_test[4][0].shape)

conv1d_train = Data(train_data, train_target, model_10,
                 max_sentence, 10, stopw, word2id, "1d")
conv1d_test = Data(test_data, test_target, model_10,
                max_sentence, 10, stopw, word2id, "1d")
print(conv1d_train[4][0].shape)
print(conv1d_test[4][0].shape)

conv2d_train = Data(train_data, train_target, model_44,
                 max_sentence, max_sentence, stopw, word2id, "2d")
conv2d_test = Data(test_data, test_target, model_44,
                max_sentence, max_sentence, stopw, word2id, "2d")
print(conv2d_train[4][0].shape)
print(conv2d_test[4][0].shape)
