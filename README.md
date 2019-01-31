# test-sentiment_analysis
*a simple test for sentiment analysis using gensim, pytorch, numpy...*

I used [this dataset](https://www.kaggle.com/vdemario/teste-de-c-digo-nlp/notebook), it's is very small for good results but is interesting about overfitting

This experiment is composed in 6 models:

* Convolutional Neural Network 2D :ok:
* Convolutional Neural Netword 1D :ok:
* Recurrent Neural Network with GRU (Gated Recurrent Unit) :ok:
* Recurrent Neural Network with LSTM (Gated Recurrent Unit) :ok:
* Convolutional Recurrent Neural Network (2d) with GRU :ok:
* Convolutional Recurrent Neural Network (1d) with GRU :ok:

### models:

```
CNN1d(
  (conv): Sequential(
    (0): Conv1d(1, 16, kernel_size=(5,), stride=(2,))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=5, stride=5, padding=0, dilation=1, ceil_mode=False)
  )
  (dropout): Dropout(p=0.2)
  (linear0): Linear(in_features=129, out_features=258, bias=True)
  (linear1): Linear(in_features=258, out_features=64, bias=True)
  (linear2): Linear(in_features=64, out_features=2, bias=True)
)

CNN2d(
  (conv0): Sequential(
    (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv1): Sequential(
    (0): Conv2d(16, 32, kernel_size=(5, 5), stride=(3, 3))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (linear): Linear(in_features=32, out_features=2, bias=True)
)

RNN (
  (embed): Embedding(4927, 10)
  (recurrence): LSTM(10, 16, num_layers=2, dropout=0.2) 
  # equal conf for GRU ^
  (linear0): Linear(in_features=704, out_features=32, bias=True)
  (linear1): Linear(in_features=32, out_features=16, bias=True)
  (linear2): Linear(in_features=16, out_features=2, bias=True)
)

CONV1dRNN(
  (embed): Embedding(4927, 10)
  (recurrence): GRU(10, 16, num_layers=2, dropout=0.2)
  (conv): Sequential(
    (0): Conv1d(1, 8, kernel_size=(5,), stride=(1,))
    (1): ReLU()
    (2): MaxPool1d(kernel_size=5, stride=5, padding=0, dilation=1, ceil_mode=False)
    (3): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (linear0): Linear(in_features=1120, out_features=32, bias=True)
  (linear1): Linear(in_features=32, out_features=16, bias=True)
  (linear2): Linear(in_features=16, out_features=2, bias=True)
)

CONV2dRNN(
  (embed): Embedding(4927, 44)
  (recurrence): GRU(44, 44, num_layers=2, dropout=0.2)
  (conv): Sequential(
    (0): Conv2d(1, 8, kernel_size=(4, 4), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=5, stride=5, padding=0, dilation=1, ceil_mode=False)
    (3): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (linear0): Linear(in_features=512, out_features=88, bias=True)
  (linear1): Linear(in_features=88, out_features=44, bias=True)
  (linear2): Linear(in_features=44, out_features=2, bias=True)
)
```



*for best visualization get [saved models](saved_models/) and use  [netron](https://www.lutzroeder.com/ai/netron/)*
