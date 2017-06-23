#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Library

# Third-party Libraries
import numpy as np

# User Define Module

# --------------------------------------------------------Global Strings----------------------------------------------------

# ------------------------------------------------------------Main----------------------------------------------------------
class Nodes(object):
    """Nodes in hidden layer of neural network.
    """
    def __init__(self, input_size, hidden_size, en_bias):
        """Initialize parameters of node.
        :Param input_size: the size of input layer
        :Param hidden_size: the size of hidden layer
        :Param en_bias: if enable bias term
        """
        self.en_bias = en_bias
        # weight matrix and bias vector
        self.u = self.random(-np.sqrt(1.0/input_size),
            np.sqrt(1.0/input_size), (hidden_size, input_size))
        self.w = self.random(-np.sqrt(1.0/hidden_size),
            np.sqrt(1.0/hidden_size), (hidden_size, hidden_size))
        self.v = self.random(-np.sqrt(1.0/hidden_size),
            np.sqrt(1.0/hidden_size), (hidden_size, hidden_size))
        if en_bias:
            self.b = self.random(-0.1, 0.1, (hidden_size,))
        else:
            self.b = np.zeros(hidden_size)
        # error gradient for weight matrix and bias vector
        self.dLdu = np.zeros(self.u.shape)
        self.dLdw = np.zeros(self.w.shape)
        self.dLdv = np.zeros(self.v.shape)
        self.dLdb = np.zeros(self.b.shape)

    def random(self, lower, upper, shape):
        """Generate a matrix whose elements are random number between lower and upper.
        :Param lower: the lower for the random number
        :Param upper: the upper for the random number
        :Param shape: the matrix's size
        """
        return np.random.uniform(lower, upper, shape)

    def reset_error(self):
        """Reset error gradient for each parameter to zero.
        """
        self.dLdu = np.zeros(self.u.shape)
        self.dLdw = np.zeros(self.w.shape)
        self.dLdv = np.zeros(self.v.shape)
        self.dLdb = np.zeros(self.b.shape)

    def update(self, alpha, beta):
        """Update nodes' parameters according to error gradient
        :Param alpha: learning rate
        :Param beta: regularization parameter
        """    
        self.u += alpha * np.clip(self.dLdu, -15, 15) - beta * self.u
        self.w += alpha * np.clip(self.dLdw, -15, 15) - beta * self.w
        self.v += alpha * np.clip(self.dLdv, -15, 15) - beta * self.v
        if self.en_bias: self.b += alpha * np.clip(self.dLdb, -15, 15) - beta * self.b

    def store(self):
        """Backup models' parameters.
        """
        self.ub = self.u.copy()
        self.wb = self.w.copy()
        self.vb = self.v.copy()
        if self.en_bias: self.bb = self.b.copy()

    def restore(self):
        """Roll back to previous iteration.
        """
        self.u = self.ub.copy()
        self.w = self.wb.copy()
        self.v = self.vb.copy()
        if self.en_bias: self.b = self.bb.copy()