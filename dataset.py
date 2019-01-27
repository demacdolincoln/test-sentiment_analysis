import torch
from torch.utils.data import Dataset
from gensim.utils import simple_preprocess
import numpy as np

def make_matrix(phrase, model_vec, m, n, stop_words):
    if type(phrase) is str:
        phrase = simple_preprocess(phrase)
        phrase = [i for i in phrase if i not in stop_words]

    out = np.zeros(m*n)
    for i, label in zip(range(0, len(phrase)*n, n), phrase):
        if label in model_vec.wv.vocab.keys():
            out[i:i+n] = model_vec[label]

    return out.reshape(1, -1)


class Data(Dataset):
    def __init__(self, data, target, model, m, n, stop_words):
        self.n = n
        self.m = m
        self.data = torch.FloatTensor(
            [make_matrix(i, model, m, n, stop_words) for i in data]
        )
        self.target = torch.LongTensor([int(i) for i in target])
        self._len = len(target)

    def __getitem__(self, x):
        return self.data[x], self.target[x]

    def __len__(self):
        return self._len
 