#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Library


# Third-party Libraries
import numpy as np

# User Define Module
from nn import NN
from nodes import Nodes
from acfun import AcFun

# --------------------------------------------------------Global Strings----------------------------------------------------
"""
Recurrent Neural Network built here can be represented as:
i(t) = sigmoid(Uix(t) + Wis(t-1) + Vic(t-1) + bi)
f(t) = sigmoid(Ufx(t) + Wfs(t-1) + Vfc(t-1) + bf)
o(t) = sigmoid(Uox(t) + Wos(t-1) + Voc(t-1) + b0)
g(t) = tanh(Ux(t) + Ws(t-1) + b)
c(t) = f(t) * c(t-1) + i(t) * g(t)
h(t) = tanh(c(t))
s(t) = o(t) * h(t)
"""

# -------------------------------------------------------------Main---------------------------------------------------------
class LSTM(NN):
    """Long Short-Term Memory Recurrent Neural Network.
    """
    def init_model(self, input_size, hidden_size, **kwargs):
        """Initialize neral network model.
        :Param input_size: size of input layer
        :Param hidden_size: size of hidden layer
        """
        # activation function for gates
        self.gatefun = AcFun(kwargs.get('GATE_FUN') or 'sigmoid')
        # parameters for input gate
        self.igate = Nodes(input_size, hidden_size, kwargs.get('EN_BIAS'))
        # parameters for forget gate
        self.fgate = Nodes(input_size, hidden_size, kwargs.get('EN_BIAS'))
        # parameters for output gate
        self.ogate = Nodes(input_size, hidden_size, kwargs.get('EN_BIAS'))
        super(LSTM, self).init_model(input_size, hidden_size, **kwargs)

    def update(self, dLds, alpha, beta):
        """Update neural network's parameters using stochastic gradient descent(SGD) method.
        :Param dLds: gradient of error in hidden layer.
        """
        T =  len(self.x)
        self.nodes.reset_error()
        self.igate.reset_error()
        self.fgate.reset_error()
        self.ogate.reset_error()
        dLdx = np.zeros((T, self.input_size))
        dLdc = np.zeros(self.hidden_size)
        for t in xrange(T-1, -1, -1):
            dLdc += dLds[t] * self.o[t] * self.acfun.derivate(self.h[t])
            dLdpi = dLdc * self.g[t] * self.gatefun.derivate(self.i[t])
            dLdpf = dLdc * self.c[t-1] * self.gatefun.derivate(self.f[t])
            dLdpo = dLds[t] * self.h[t] * self.gatefun.derivate(self.o[t])
            dLdpg = dLdc * self.i[t] * self.acfun.derivate(self.g[t])
            dLdc = dLdc * self.f[t]
            # parameters for nodes in hidden layer
            self.nodes.dLdu += np.outer(dLdpg, self.x[t])
            self.nodes.dLdw += np.outer(dLdpg, self.s[t-1])
            dLds[t-1] += np.dot(self.nodes.w.T, dLdpg)
            dLdx[t] += np.dot(self.nodes.u.T, dLdpg)
            # parameters for input gate
            self.igate.dLdu += np.outer(dLdpi, self.x[t])
            self.igate.dLdw += np.outer(dLdpi, self.s[t-1])
            self.igate.dLdv += np.outer(dLdpi, self.c[t-1])
            dLds[t-1] += np.dot(self.igate.w.T, dLdpi)
            dLdx[t] += np.dot(self.igate.u.T, dLdpi)
            dLdc += np.dot(self.igate.v.T, dLdpi)
            # parameters for forget gate
            self.fgate.dLdu += np.outer(dLdpf, self.x[t])
            self.fgate.dLdw += np.outer(dLdpf, self.s[t-1])
            self.fgate.dLdv += np.outer(dLdpf, self.c[t-1])
            dLds[t-1] += np.dot(self.fgate.w.T, dLdpf)
            dLdx[t] += np.dot(self.fgate.u.T, dLdpf)
            dLdc += np.dot(self.fgate.v.T, dLdpf)
            # parameters for output gate
            self.ogate.dLdu += np.outer(dLdpo, self.x[t])
            self.ogate.dLdw += np.outer(dLdpo, self.s[t-1])
            self.ogate.dLdv += np.outer(dLdpo, self.c[t-1])
            dLds[t-1] += np.dot(self.ogate.w.T, dLdpo)
            dLdx[t] += np.dot(self.ogate.u.T, dLdpo)
            dLdc += np.dot(self.ogate.v.T, dLdpo)
            if self.en_bias:
                self.nodes.dLdb += dLdpg
                self.igate.dLdb += dLdpi
                self.fgate.dLdb += dLdpf
                self.ogate.dLdb += dLdpo
        # update weight matrix of current hidden node
        self.nodes.update(alpha, beta)
        self.igate.update(alpha, beta)
        self.fgate.update(alpha, beta)
        self.ogate.update(alpha, beta)
        return dLdx
    
    def run(self, x):
        """Forward propagation, calculate the network for given input.
        :Param x: input sequence
        """
        T = len(x)
        self.x = x
        self.i = np.zeros((T, self.hidden_size))
        self.f = np.zeros((T, self.hidden_size))
        self.o = np.zeros((T, self.hidden_size))
        self.g = np.zeros((T, self.hidden_size))
        self.h = np.zeros((T, self.hidden_size))
        self.c = np.zeros((T+1, self.hidden_size))
        self.s = np.zeros((T+1, self.hidden_size))
        for t in xrange(T):
            # input gate
            self.i[t] = self.gatefun.compute(np.dot(self.igate.u, x[t])
                + np.dot(self.igate.w, self.s[t-1])
                + np.dot(self.igate.v, self.c[t-1]) + self.igate.b)
            # forget gate
            self.f[t] = self.gatefun.compute(np.dot(self.fgate.u, x[t])
                + np.dot(self.fgate.w, self.s[t-1])
                + np.dot(self.fgate.v, self.c[t-1]) + self.fgate.b)
            # output gate
            self.o[t] = self.gatefun.compute(np.dot(self.ogate.u, x[t])
                + np.dot(self.ogate.w, self.s[t-1])
                + np.dot(self.ogate.v, self.c[t-1]) + self.ogate.b)
            # current hidden node state
            self.g[t] = self.acfun.compute(np.dot(self.nodes.u, x[t]) + 
                np.dot(self.nodes.w, self.s[t-1]) + self.nodes.b)
            # internal memoery
            self.c[t] = self.f[t] * self.c[t-1] + self.i[t] * self.g[t]
            self.h[t] = self.acfun.compute(self.c[t])
            self.s[t] = np.clip(self.o[t] * self.h[t], -50, 50)
        return self.s[:-1]
    
    def store(self):
        """Backup models' parameters.
        """
        self.igate.store()
        self.fgate.store()
        self.ogate.store()
        super(LSTM, self).store()

    def restore(self):
        """Roll back to previous iteration.
        """
        self.igate.restore()
        self.fgate.restore()
        self.ogate.restore()
        super(LSTM, self).restore()