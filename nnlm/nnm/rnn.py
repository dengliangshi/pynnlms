#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Library

# Third-party Libraries
import numpy as np

# User Define Module
from nn import NN

# --------------------------------------------------------Global Strings----------------------------------------------------
"""
Recurrent Neural Network built here can be represented as:
s(t) = f(Ux(t) + Ws(t-1) + b)
"""

# ----------------------------------------------------------Class RNN-------------------------------------------------------
class RNN(NN):
    """Recurrent Neural Network.
    """
    def update(self, dLds, alpha, beta):
        """Update neural network's parameters using stochastic gradient descent(SGD) method.
        :Param dLds: gradient of error in hidden layer.
        :Param alpha: learning rate
        :Param beta: regularization parameter
        """
        T =  len(self.x)
        dLdx = np.zeros((T, self.input_size))
        self.nodes.reset_error()
        for t in xrange(T-1, -1, -1):
            dLdp = dLds[t] * self.acfun.derivate(self.s[t])
            self.nodes.dLdu += np.outer(dLdp, self.x[t])
            dLdx[t] += np.dot(self.nodes.u.T, dLdp)
            if self.en_bias: self.nodes.dLdb += dLdp
            self.nodes.dLdw += np.outer(dLdp, self.s[t-1])
            dLds[t-1] += np.dot(self.nodes.w.T, dLdp)
        self.nodes.update(alpha, beta)
        return dLdx
    
    def run(self, x):
        """Forward propagation, calculate the network for given input.
        :Param x: input vectors
        """
        T = len(x)
        self.x = x
        self.s = np.zeros((T+1, self.hidden_size))
        for t in xrange(T):
            self.s[t] = np.clip(np.dot(self.nodes.u, x[t])
                + np.dot(self.nodes.w, self.s[t-1]) + self.nodes.b, -50, 50)
            self.s[t] = self.acfun.compute(self.s[t])
        return self.s[:-1]