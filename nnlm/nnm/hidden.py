#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Libraries

# Third-party Libraries
import numpy as np

# User Define Modules
from fnn import FNN
from rnn import RNN
from lstm import LSTM
from birnn import BiRNN

# --------------------------------------------------------Global Strings----------------------------------------------------

# -------------------------------------------------------------Main---------------------------------------------------------
class Hidden(object):
    """Hidden layers of neural network model.
    """
    models = {
        'FNN': FNN,
        'RNN': RNN,
        'LSTM': LSTM,
        'BiRNN': BiRNN,
        'BiLSTM': BiRNN
    }
    def __init__(self):
        """Initialization Function.
        """
        # hidden layers of neural network model
        self.layers = []
        # the size of input layer of nerual network model
        self.input_size = 0
        # the size of last hidden layer
        self.output_size = 0
        # the number of hidden layers
        self.layer_num = 0

    def __repr__(self):
        """Instance display format.
        """
        pass

    def init_model(self, config):
        """Initialize language model.
        :Param config: configuration of neural network model.
        """
        # get configuration parameters for neural network model
        self.vector_dim = config.get('VECTOR_DIM')
        self.gram_order = config.get('GRAM_ORDER')
        # initialize hidden layers
        hidden_layers = config.get('HIDDEN_LAYERS')
        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers, ]
        if hidden_layers[0][0] == 'FNN':
            self.input_size = self.vector_dim*(self.gram_order-1)
        else:
            self.input_size = self.vector_dim
        input_size = self.input_size
        self.layer_num = len(hidden_layers)
        for name, hidden_size in hidden_layers:
            self.layers.append(self.models[name](name))
            self.layers[-1].init_model(input_size, hidden_size, **config)
            input_size = hidden_size
        self.output_size = input_size
        return self.input_size, self.output_size

    def run(self, x):
        """Run neural network over given input.
        :Param x: input vectors
        """
        for index in xrange(self.layer_num):
            x = self.layers[index].run(x)
        return x
    
    def update(self, dLds, alpha, beta):
        """Update word's feature vector.
        :Param dLds: gradient of error
        :Param alpha: learning rate
        :Param beta: regularization parameter
        """
        for index in xrange(self.layer_num-1, -1, -1):
            dLds = self.layers[index].update(dLds, alpha, beta)
        return dLds

    def store(self):
        """Backup models' parameters.
        """
        for index in xrange(self.layer_num):
            self.layers[index].store()

    def restore(self):
        """Roll back to previous iteration.
        """
        for index in xrange(self.layer_num):
            self.layers[index].restore()
