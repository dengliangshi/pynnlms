#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Libraries
import os

# Third-party Libraries


# User Defined Modules
from nnlm import NNLM

# --------------------------------------------------------Global Strings----------------------------------------------------
root = os.path.dirname(__file__)  # the root path of working directory

# -------------------------------------------------------------Main---------------------------------------------------------
config = {
    "MODEL_NAME": 'LSTMLM',                                # name for this language model
    "TRAIN_FILES": os.path.join(root, 'input/train'),      # the parent directary of training file(s)
    "VALID_FILES": os.path.join(root, 'input/valid'),      # the parent directary of validation file(s)
    "TEST_FILES": os.path.join(root, 'input/test'),        # the parent directary of test file(s)
    "OUTPUT_PATH": os.path.join(root, 'output'),           # the path for saving output files
    "HIDDEN_LAYERS": ('LSTM', 30),                         # hidden layers of nerual network
    "FILE_TYPE": 'T',                                      # the type of input files, 'T' for text and 'B' for binary
    "VOCAB_SIZE": 100000,                                  # the size of vocabulary
    "WORD_CLASS": 100,                                     # the number of word class
    "INPUT_UNIT": 'W',                                     # the input level, W for word and C for character
    "VECTOR_DIM": 30,                                      # the size of feature vector
    "RANDOM_SEED": 1,                                      # seed for random generation
    "ALPHA": 0.02,                                         # learning rate of gradient descent algorithm
    "BETA": 1.0e-6,                                        # regularization parameter
    "ITERATIONS": 20,                                      # maximal iterations for training
    "MIN_IMPROVE": 1.003,                                  # the minimum rate of entropy improvement on validation data
    "AC_FUN": 'tanh',                                      # activation function in hidden layer
    'GATE_FUN': 'sigmoid',                                 # activation function for gates in LSTM
    "EN_DIRECT": False,                                    # if using direct connection between input and output layer in RNN
    "EN_BIAS": False,                                      # if using bias in RNN
    "ALPHA_CUT": 0.5,                                      # ratio for the cutoff of alpha when small improvement found
}

nnlm = NNLM()
nnlm.config.from_dict(config)
nnlm.init_model()
nnlm.train()
nnlm.test()
nnlm.save()