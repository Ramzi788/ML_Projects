%%writefile basic_rnn_cell.py
from torch import nn, Tensor, tanh
from math import sqrt


class BasicRNNCell(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BasicRNNCell, self).__init__()
        """
        Creates an RNN cell with a tanH activation function

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in the rnn cell.

        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # create and initialize parameters W, V, b as described in the text.
        # remember that the parameters are instance variables

        # W, the input weights matrix has size (n x m) where n is
        # the number of input features and m is the hidden size
        # V, the hidden state weights matrix has size (m, m)
        # b, the vector of bias, has size (m)

        n = vocab_size
        m = hidden_size
        k = sqrt(1 / m)

        self.W = nn.Parameter(Tensor(n, m).uniform_(-k, k))
        self.V = nn.Parameter(Tensor(m, m).uniform_(-k, k))
        self.b = nn.Parameter(Tensor(m).uniform_(-k, k))

    def forward(self, x, h):
        """
        Defines the forward propagation of an RNN cell with a tanH as activation function

        Arguments
        ---------
        x: (Tensor) of size (B x n) where B is the mini-batch size and n is the number of input features. x is
            the input data of the current time-step. In a multi-layer RNN, x is the previous layer's hidden state
        h: (Tensor) of size (B x m) where m is the hidden size. h is the hidden state of the previous layer

        Return
        ------
        h: (Tensor) of size (B x m), the new hidden state

        """

        a = self.b + x @ self.W + h @ self.V
        h = tanh(a)

        return h