import torch
from torch import nn

class CNN1d(nn.Module):
    def __init__(self, lin_in, lin_out):
        super(CNN1d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5)
        )

        self.dropout = nn.Dropout(p=0.2)

        self.linear0 = nn.Linear(lin_in, int(lin_in*2))
        self.linear1 = nn.Linear(int(lin_in*2), int(lin_in/2))
        self.linear2 = nn.Linear(int(lin_in/2), lin_out)

    def forward(self, x):
        x = self.conv(x)

        x = x.view(x.shape[0], -1)
        x = self.linear0(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = torch.sigmoid(self.linear2(x))
        return x


class CNN2d(nn.Module):
    def __init__(self, lin_in, lin_out):
        super(CNN2d, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.linear = nn.Linear(lin_in, lin_out)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        x = torch.sigmoid(self.linear(x))
        return x


class RNN(nn.Module):
    def __init__(self, vocab_size, model, m, n, n_layers, hidden, out, mode):
        super(RNN, self).__init__()

        self.n_layers = n_layers
        self.m = m
        self.n = n
        self.hidden_size = hidden
        self.mode = mode

        self.embed = nn.Embedding(vocab_size, n)
        self.embed.load_state_dict({
            "weight": torch.FloatTensor(model.wv.vectors)
        })

        if mode is "gru":
            self.recurrence = nn.GRU(n, hidden, n_layers, dropout=0.2)
        elif mode is "lstm":
            self.recurrence = nn.LSTM(n, hidden, n_layers, dropout=0.2)
        else:
            raise "escolha entre gru e lstm apenas"

        self.linear0 = nn.Linear(m*hidden, int(hidden*2))
        self.linear1 = nn.Linear(int(hidden*2), hidden)
        self.linear2 = nn.Linear(hidden, out)

    def forward(self, inpt):

        hidden = self._init_hidden(inpt.size(1))

        x, hidden = self.recurrence(
            self.embed(inpt),
            hidden
        )

        space = torch.zeros(self.m*self.hidden_size)
        x = x.flatten()
        space[:x.shape[0]] = x
        x = space

        x = self.linear0(x)
        x = self.linear1(x)
        x = self.linear2(x)

        return torch.sigmoid(x).unsqueeze(0)

    def _init_hidden(self, batch_size):
        if self.mode == "lstm":
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size))
        else:
            return torch.zeros(self.n_layers, batch_size, self.hidden_size)


class CONV1dRNN(nn.Module):
    def __init__(self, vocab_size, model, m, n, n_layers, hidden, out, mode):
        super(CONV1dRNN, self).__init__()

        self.n_layers = n_layers
        self.m = m
        self.n = n
        self.hidden_size = hidden
        self.mode = mode

        self.embed = nn.Embedding(vocab_size, n)
        self.embed.load_state_dict({
            "weight": torch.FloatTensor(model.wv.vectors)
        })

        if mode is "gru":
            self.recurrence = nn.GRU(n, hidden, n_layers, dropout=0.2)
        elif mode is "lstm":
            self.recurrence = nn.LSTM(n, hidden, n_layers, dropout=0.2)
        else:
            raise "escolha entre gru e lstm apenas"

        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, 5),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.BatchNorm1d(8)
        )

        self.linear0 = nn.Linear(1120, int(hidden*2))
        self.linear1 = nn.Linear(int(hidden*2), hidden)
        self.linear2 = nn.Linear(hidden, out)

    def forward(self, inpt):

        hidden = self._init_hidden(inpt.size(1))

        x, hidden = self.recurrence(
            self.embed(inpt),
            hidden
        )

        space = torch.zeros(1, 1, self.m*self.hidden_size)
        x = x.flatten()
        space[0, 0, :x.shape[0]] = x
        x = space
        x = self.conv(x)
        x = x.flatten()
        x = self.linear0(x)
        x = self.linear1(x)
        x = self.linear2(x)

        return torch.sigmoid(x).unsqueeze(0)

    def _init_hidden(self, batch_size):
        if self.mode == "lstm":
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size))
        else:
            return torch.zeros(self.n_layers, batch_size, self.hidden_size)


class CONV2dRNN(nn.Module):
    def __init__(self, vocab_size, model, m, n, n_layers, hidden, out, mode):
        super(CONV2dRNN, self).__init__()

        self.n_layers = n_layers
        self.m = m
        self.n = n
        self.hidden_size = hidden
        self.mode = mode

        self.embed = nn.Embedding(vocab_size, m)
        self.embed.load_state_dict({
            "weight": torch.FloatTensor(model.wv.vectors)
        })

        if mode is "gru":
            self.recurrence = nn.GRU(n, hidden, n_layers, dropout=0.2)
        elif mode is "lstm":
            self.recurrence = nn.LSTM(n, hidden, n_layers, dropout=0.2)
        else:
            raise "escolha entre gru e lstm apenas"

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(5),
            nn.BatchNorm2d(8)
        )

        self.linear0 = nn.Linear(512, int(hidden*2))
        self.linear1 = nn.Linear(int(hidden*2), hidden)
        self.linear2 = nn.Linear(hidden, out)

    def forward(self, inpt):

        hidden = self._init_hidden(inpt.size(1))

        x, hidden = self.recurrence(
            self.embed(inpt),
            hidden
        )
        
        space = torch.zeros(1, 1, self.m, self.hidden_size)

        space[0, 0, :x.shape[1], :] = x
        x = space
        x = self.conv(x)
        x = x.flatten()
        x = self.linear0(x)
        x = self.linear1(x)
        x = self.linear2(x)

        return torch.sigmoid(x).unsqueeze(0)

    def _init_hidden(self, batch_size):
        if self.mode == "lstm":
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size))
        else:
            return torch.zeros(self.n_layers, batch_size, self.hidden_size)
