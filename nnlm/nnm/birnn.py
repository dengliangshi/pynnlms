#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Library

# Third-party Libraries
import numpy as np

# User Define Module
from rnn import RNN
from lstm import LSTM

# --------------------------------------------------------Global Strings----------------------------------------------------


# -------------------------------------------------------------Main---------------------------------------------------------
class BiRNN(object):
    """Bidirectional Recurrent Neural Network.
    """
    models = {
        'BiRNN': RNN,
        'BiLSTM': LSTM
    }
    def __init__(self, name):
        """Initialize function.
        :Param name: the name of this neural network model.
        """
        self.name = name
        # forward rnn model
        self.frnn = self.models[name](name)
        # backward rnn model
        self.brnn = self.models[name](name)

    def __repr__(self):
        """Instance display format.
        """
        return '<Neural Network: %s>' % self.name

    def init_model(self, input_size, hidden_size, **kwargs):
        """Initialize neral network model.
        :Param input_size: size of input layer
        :Param hidden_size: size of hidden layer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.half_hidden_size = hidden_size / 2 
        self.frnn.init_model(input_size, self.half_hidden_size, **kwargs)
        self.brnn.init_model(input_size, self.half_hidden_size, **kwargs)

    def run(self, x):
        """Run neural network over given input.
        :Param x: input vectors
        """
        T = len(x)
        fs = self.frnn.run(x)
        bs = self.brnn.run(x[::-1])
        s = np.zeros((T, self.hidden_size))
        for t in xrange(T):
            s[t] = np.concatenate((fs[t], bs[T-t-1]))
        return s
    
    def update(self, dLds, alpha, beta):
        """Update word's feature vector.
        :Param dLds: gradient of error
        :Param alpha: learning rate
        :Param beta: regularization parameter
        """
        T = len(dLds)
        dLdfs = np.zeros((T, self.half_hidden_size))
        dLdbs = np.zeros((T, self.half_hidden_size))
        for t in xrange(T):
            dLdfs[t] = dLds[t][:self.half_hidden_size]
            dLdbs[T-t-1] = dLds[t][self.half_hidden_size:]
        dLdfx = self.frnn.update(dLdfs, alpha, beta)
        dLdbx = self.brnn.update(dLdbs, alpha, beta)
        return dLdfx + dLdbx