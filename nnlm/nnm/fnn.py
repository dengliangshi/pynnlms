
#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Library

# Third-party Libraries
import numpy as np

# User Define Module
from nn import NN

# --------------------------------------------------------Global Strings----------------------------------------------------
"""Feed-Forward Neural Network built here can be represented as:
s = f(Ux + b)
"""

# -------------------------------------------------------------Main---------------------------------------------------------
class FNN(NN):
    """Feed-forward neural network.
    """
    def update(self, dLds, alpha, beta):
        """Update neural network's parameters using stochastic gradient descent(SGD) method.
        :Param dLds: error gradients of hidden layer's outputs.
        :Param alpha: learning rate
        :Param beta: regularization parameter
        """
        T = len(self.x)
        dLdx = np.zeros((T, self.input_size))
        self.nodes.reset_error()
        for t in xrange(T):
            dLdp = dLds[t] * self.acfun.derivate(self.s[t])
            self.nodes.dLdu += np.outer(dLdp, self.x[t])
            if self.en_bias: self.nodes.dLdb += dLdp
            dLdx[t] = np.dot(self.nodes.u.T, dLdp)
        self.nodes.update(alpha, beta)
        return dLdx
    
    def run(self, x):
        """Forward propagation, calculate the network for given input.
        :Param x: input sequence
        """
        T = len(x)
        self.x = x
        self.s = np.zeros((T+1, self.hidden_size))
        for t in xrange(T):
            self.s[t] = np.dot(self.nodes.u, x[t]) + self.nodes.b
            self.s[t] = self.acfun.compute(np.clip(self.s[t], -50, 50))
        return self.s