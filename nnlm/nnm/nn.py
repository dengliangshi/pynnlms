#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Library

# Third-party Libraries
import numpy as np

# User Define Module
from nodes import Nodes
from acfun import AcFun

# --------------------------------------------------------Global Strings----------------------------------------------------

# -------------------------------------------------------------Main---------------------------------------------------------
class NN(object):
    """Meta class for neural network models.
    """
    def __init__(self, name):
        """Initialize function.
        :Param name: the name of this neural network model.
        """
        self.name = name

    def __repr__(self):
        """Instance display format.
        """
        return '<Neural Network: %s, size: %d>' % (self.name, self.hidden_size)

    def init_model(self, input_size, hidden_size, **kwargs):
        """Initialize neral network model.
        :Param input_size: size of input layer
        :Param hidden_size: size of hidden layer
        :Param en_bptt: if using BPTT for calculating error gradients
        """
        # the size of input layer
        self.input_size = input_size
        # the node number of hidden layer
        self.hidden_size = hidden_size
        # if use bias term in activation function in hidden layer
        self.en_bias = kwargs.get('EN_BIAS') or False
        # activation function in hidden layer
        self.acfun = AcFun(kwargs.get('AC_FUN') or 'tanh')
        # parameters for nodes in hidden layer
        self.nodes = Nodes(self.input_size, self.hidden_size, self.en_bias)

    def random(self, lower, upper, shape):
        """Generate a matrix whose elements are random number between lower and upper.
        :Param lower: the lower for the random number
        :Param upper: the upper for the random number
        :Param shape: the matrix's size
        """
        return np.random.uniform(lower, upper, shape)

    def store(self):
        """Backup models' parameters.
        """
        self.nodes.store()

    def restore(self):
        """Roll back to previous iteration.
        """
        self.nodes.restore()
        