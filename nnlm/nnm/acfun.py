#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Library

# Third-party Libraries
import numpy as np

# User Define Module

# --------------------------------------------------------Global Strings----------------------------------------------------

# -------------------------------------------------------------Main---------------------------------------------------------
class AcFun(object):
    """Activation functions for neural network models
    """
    def __init__(self, name):
        """Initialize the activation function with the given function name.
        """
        functions = {
            'tanh': Tanh,
            'relu': ReLu,
            'sigmoid': Sigmoid,
            'hardsigmoid': HardSigmoid,
        }
        function = functions.get(name)
        if functions is None:
            print 'Function %s does not exist!' % name
        else:
            self.acfun = function()
    
    def __repr__(self):
        """Instance display format.
        """
        return str(self.acfun)

    def compute(self, x):
        """Run tanh for given input and save the corresponding derivate.
        :Param x: the input of function, a single number or matrix.
        """
        return self.acfun.compute(x)

    def derivate(self, y):
        """Return derivate.
        :Param y: the output of function.
        """
        return self.acfun.derivate(y)


class Tanh(object):
    """Tanh Function.
    """
    def __repr__(self):
        """Instance display format.
        """
        return '<Function Tanh>'

    def compute(self, x):
        """Run tanh for given input and save the corresponding derivate.
        :Param x: the input of function, a single number or matrix.
        """
        return np.tanh(x)

    def derivate(self, y):
        """Return derivate.
        :Param y: the output of function
        """
        return 1.0 - y ** 2


class Sigmoid(object):
    """Sigmoid Function.
    """
    def __repr__(self):
        """Instance display format.
        """
        return '<Function Sigmoid>'

    def compute(self, x):
        """Run tanh for given input and save the corresponding derivate.
        :Param x: the input of function, a single number or matrix.
        """
        return 1.0 / (1.0 + np.exp(-x))

    def derivate(self, y):
        """Return derivate.
        :Param y: the output of function
        """
        return y * (1.0 - y)


class HardSigmoid(object):
    """Hard Sigmoid Function.
    """
    def __repr__(self):
        """Instance display format.
        """
        return '<Function Hard Sigmoid>'

    def compute(self, x):
        """Run Hard Sigmoid for given input and save the corresponding derivate.
        :Param x: the input of function, a single number or matrix.
        """
        return np.maximum(0, np.minimum(1.0, (x + 1.0)/2.0)) 

    def derivate(self, y):
        """Return derivate.
        :Param y: the output of function
        """
        return np.where((y > 0) & (y < 1.0), 0.5, 0)


class ReLu(object):
    """ReLu Function.
    """
    def __repr__(self):
        """Instance display format.
        """
        return '<Function ReLu>'

    def compute(self, x):
        """Run ReLu for given input and save the corresponding derivate.
        :Param x: the input of function, a single number or matrix.
        """
        return np.maximum(0, x)

    def derivate(self, y):
        """Return derivate.
        :Param y: the output of function
        """
        return np.where((y > 0), 1.0, 0)

class Gaussian(object):
    """Gaussian Function.
    """
    def __init__(self):
        """Initialization Function.
        """
        self.dydx = None

    def __repr__(self):
        """Instance display format.
        """
        return '<Function Gaussian>'

    def compute(self, x, a, b):
        """Run Gaussian for given input and save the corresponding derivate.
        :Param x: the input of function, a single number or matrix.
        """
        return np.exp(-np.power((x-a), 2) / (2.0 * np.power(b, 2)))

    def derivate(self, y):
        """Return derivate.
        :Param y: the output of function
        """
        return np.where((y > 0), 1.0, 0)