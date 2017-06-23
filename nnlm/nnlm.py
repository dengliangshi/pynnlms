#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Libraries
import os
import time
import codecs
import cPickle as pickle
from glob import glob

# Third-party Libraries
import numpy as np

# User Define Modules
from nnm import Hidden
from vocab import Vocab
from config import Config

# --------------------------------------------------------Global Strings----------------------------------------------------


# -------------------------------------------------------------Main---------------------------------------------------------
class NNLM(object):
    """Neural Network Language Model.
    """
    # default configuration parameters
    default_config = {
        'FILE_TYPE': 'T',              # the type of input files, 'T' for text and 'B' for binary
        'VOCAB_SIZE': 10000,           # the size of vocabulary
        'CLASS_SIZE': 100,             # number of word class
        'INPUT_UNIT': 'W',             # the input level, W for word and C for character
        'VECTOR_DIM': 30,              # the size of feature vector
        'RANDOM_SEED': 1,              # seed for random generation
        'ALPHA': 0.01,                 # learning rate of gradient descent algorithm
        'BETA': 1e-6,                  # regularization parameter
        'ITERATIONS': 20,              # maximal iterations for training
        'AC_FUN': 'tanh',              # activation function for hidden layer
        'GATE_FUN': 'sigmoid',         # activation function for gates in LSTM
        'EN_DIRECT': False,            # if using direct connection between input and output layer
        'EN_BIAS': False,              # if using bias
        'GRAM_ORDER': 4,               # the order of for FNN
        'HIDDEN_LAYERS': ('FNN', 30),  # hidden layers of nerual network
        'SENTENCE_START': '<s>',       # flag fot the start of a sentence
        'SENTENCE_END': '<\s>',        # flag for the end of a sentence
        'UNKNOWN_WORD': 'OOV',         # word out of vocabulary
        'ALPHA_CUT': 0.75,             # ratio for cutting alpha when small improvement found
        'MIN_IMPROVE': 1.003,          # the minimum rate of entropy improvement on validation data
    }

    def __init__(self):
        """Initialization Function.
        """
        # configuration of neural network language model
        self.config = Config(self.default_config)
        # instance of vocabulary
        self.vocab = Vocab()
        # instance of neural network language model
        self.hidden = Hidden()
        # previous entropy on validation data set
        self.pre_entropy = float('inf')
        # flag for learning rate turning
        self.adjust_alpha = False
        # if the first hidden layer is feed-forward
        self.is_fnn = False
        # word counter
        self.word_num = 0
        # logarithm probabilty
        self.logp = 0

    def pre_check(self):
        """check the given parameters.
        """
        # check if the name of language model is ok
        self.model_name = self.config.get('MODEL_NAME')
        if not self.model_name:
            raise Exception('Please give a name for this language model!')
        # check if training files are given in a proper way
        self.train_files = self.config.get('TRAIN_FILES')
        if self.train_files is None or not os.path.isdir(self.train_files):
            raise Exception('Training files is not given or not a directory!')
        # check if validation files are given in a proper way
        self.valid_files = self.config.get('VALID_FILES')
        if self.valid_files is None or not os.path.isdir(self.valid_files):
            raise Exception('Validation files is not given or not a directory!')
        # check if test files are given in a proper way
        self.test_files = self.config.get('TEST_FILES')
        if self.test_files is None or not os.path.isdir(self.test_files):
            raise Exception('Test files is not given or not a directory!')
        # check if test files are given in a proper way
        self.output_path =  self.config.get('OUTPUT_PATH')
        if self.output_path is None or not os.path.isdir(self.output_path):
            raise Exception('Path for output files is not given or not a directory!')
        # check if the given file type is ok
        self.file_type = self.config.get('FILE_TYPE')
        if self.file_type not in ['T', 'B']:
            raise Exception('The supported file type is T(text) or B(binary)!')
        self.file_mode = 'r' if self.file_type == 'T' else 'rb'
        # check if the size of word class is ok
        self.class_size = self.config.get('CLASS_SIZE')
        if not isinstance(self.class_size, int) or self.class_size < 1:
            raise Exception('The size of word class should be a positive integer!')
        # check if the dimension of feature vector for words is ok
        self.vector_dim = self.config.get('VECTOR_DIM')
        if not isinstance(self.vector_dim, int) or self.vector_dim < 1:
            raise Exception('The dimension of feature vector for words should be a positive integer!')
        # check if the grammar order is ok
        self.gram_order = self.config.get('GRAM_ORDER')
        if not isinstance(self.gram_order, int) or self.gram_order < 1:
            raise Exception('The grammar order should be a positive integer!')
        # check if the learning rate is ok
        self.alpha = self.config.get('ALPHA')
        if not isinstance(self.alpha, float) or self.alpha < 0:
            raise Exception('The learning rate should be a positive number!')
        # check if the regularization parameter is ok
        self.beta = self.config.get('BETA')
        if not isinstance(self.beta, float) or self.beta < 0:
            raise Exception('The learning rate should be a positive number!')
        # check if the minimum improvement rate is ok
        self.min_improve = self.config.get('MIN_IMPROVE')
        if not isinstance(self.min_improve, float) or self.min_improve < 1:
            raise Exception('The minimum improvement rate should be greater than 1!')
        # check if the maximum of iteration is ok
        self.iter_num = self.config['ITERATIONS']
        if not isinstance(self.iter_num, int) or  self.iter_num < 1:
            raise Exception('The maximum iteration should be a postive integer!')
        # check if the input level is ok
        self.input_unit = self.config.get('INPUT_UNIT')
        if self.input_unit not in ['W', 'C']:
            raise Exception('The input level should be "W" for word or "C" for character!')
        # check if the mark for the start of sentence is ok
        self.sentence_start = self.config.get('SENTENCE_START')
        if not isinstance(self.sentence_start, str):
            raise Exception('The mark for the start of sentence should be a string!')
        # check if the mark for the end of sentence is ok
        self.sentence_end = self.config.get('SENTENCE_END')
        if not isinstance(self.sentence_end, str):
            raise Exception('The mark for the end of sentence should be a string!')
        # check if the mark for out of vocabulary words is ok
        self.unknown_word = self.config.get('UNKNOWN_WORD')
        if not isinstance(self.unknown_word, str):
            raise Exception('The mark for the out of vocabulary words should be a string!')
        # check if the flag for enabling direct connections is ok
        self.en_direct = self.config.get('EN_DIRECT')
        if not isinstance(self.en_direct, bool):
            raise Exception('Only True or False should be given to enable or disable direct connections!')
        # check if the flag for enabling bias terms is ok
        self.en_bias = self.config.get('EN_BIAS')
        if not isinstance(self.en_bias, bool):
            raise Exception('Only True or False should be given to enable or disable bias terms!')
        # check if the flag for enabling bias terms is ok
        self.alpha_cut = self.config.get('ALPHA_CUT')
        if not isinstance(self.alpha_cut, float) or self.alpha_cut >= 1 or self.alpha_cut <= 0:
            raise Exception('The rate for cut alpha should a float between 0 and 1!')
        # check if the seed for random generater is ok
        self.random_seed = self.config.get('RANDOM_SEED')
        if not isinstance(self.random_seed, int) or self.random_seed < 1:
            raise Exception('The seed for random generater should be a postive integer!')

    def init_model(self):
        """Initialize language model.
        """
        self.pre_check()
        # initialize random seed
        np.random.seed(self.random_seed)
        # initialize vocabulary
        self.vocab.init_model(self.config)
        self.vocab_size = self.vocab.generate()
        # initialize neural network model
        self.input_size, self.hidden_size = self.hidden.init_model(self.config)
        # if the first hidden layer is feedforward nerual network
        if self.input_size != self.vector_dim: self.is_fnn = True
        # projection matrix, feature vectors for words, retrieved by word index
        self.C = self.random(-0.1, 0.1, (self.vocab_size, self.vector_dim))
        word = self.vocab.get_word(self.unknown_word)
        self.C[word.index] = np.zeros(self.vector_dim)
        # initialize weight matrix for output layer
        self.V = self.random(-np.sqrt(1.0/self.hidden_size),
            np.sqrt(1.0/self.hidden_size), (self.vocab_size, self.hidden_size))
        self.Vc = self.random(-np.sqrt(1.0/self.hidden_size),
            np.sqrt(1.0/self.hidden_size), (self.class_size, self.hidden_size))
        # initialize weight for direct connections
        if self.en_direct:
            self.M = self.random(-np.sqrt(1.0/self.input_size),
                np.sqrt(1.0/self.input_size), (self.vocab_size, self.input_size))
            self.Mc = self.random(-np.sqrt(1.0/self.input_size),
                np.sqrt(1.0/self.input_size), (self.class_size, self.input_size))
        else:
            self.M = np.zeros((self.vocab_size, self.input_size))
            self.Mc = np.zeros((self.class_size, self.input_size))
        # initialize bias vectors
        if self.en_bias:
            self.d = self.random(-0.1, 0.1, (self.vocab_size,))
            self.dc = self.random(-0.1, 0.1, (self.class_size,))
        else:
            self.d = np.zeros(self.vocab_size)
            self.dc = np.zeros(self.class_size)
        self.output_file = os.path.join(self.output_path, self.model_name)
        self.pre_save()
        

    def pre_save(self):
        """Save the parameters of this language model.
        """
        with codecs.open(self.output_file+'.txt', 'w', encoding='utf-8') as output:
            output.write('Model Name: %s\n' % self.model_name);
            output.write('Training Files: %s\n' % self.train_files);
            output.write('Validation Files: %s\n' % self.valid_files);
            output.write('Test Files: %s\n' % self.test_files);
            output.write('Output Path: %s\n' % self.output_path)
            output.write('File Type: %s\n' % ('Text' if self.file_type == 'T' else 'Binary'))
            output.write('Vocabulary Size: %d\n' % self.vocab_size)
            output.write('NUmber of Word Class: %d\n' % self.class_size)
            output.write('Size of Feature Vectors: %d\n' % self.vector_dim)
            output.write('Grammar Order: %d\n' % self.gram_order)
            output.write('Initial Learning Rate: %.2f\n' % self.alpha)
            output.write('Regularization Parameter: %.9f\n' % self.beta)
            output.write('Minimum Improvement rate: %.5f\n' % self.min_improve)
            output.write('Maximun Interation: %d\n' % self.iter_num)
            output.write('Input Level: %s\n' % ('Word' if self.input_unit == 'W' else 'Character'))
            output.write('Mark for Start of Sentence: %s\n' % self.sentence_start)
            output.write('Mark for End of Sentence: %s\n' % self.sentence_end)
            output.write('Mark for Words Out of Vocabulary: %s\n' % self.unknown_word)
            output.write('Enable Direct Connections: %r\n' % self.en_direct)
            output.write('Enable Bias Terms: %r\n' % self.en_bias)
            output.write('Rate for Cutoff Learning Rate: %d\n' % self.alpha_cut)
            output.write('Random Seed: %d\n' % self.random_seed)

    def random(self, lower, upper, size):
        """Generate a vector whose elements are random numbers between min and max.
        :Param lower: the lower for the random number
        :Param upper: the upper for the random number
        :Param size: the size of vector
        """
        return np.random.uniform(lower, upper, size)

    def softmax(self, x):
        """Softmax function.
        Param x: the input of function, a vector.
        """
        return np.exp(x) / np.sum(np.exp(x))

    def restore(self):
        """Roll back to previous interation step.
        """
        self.C = self.Cb.copy()
        self.V = self.Vb.copy()
        self.Vc = self.Vcb.copy()
        if self.en_direct:
            self.M = self.Mb.copy()
            self.Mc = self.Mcb.copy()
        if self.en_bias:
            self.d = self.db.copy()
            self.dc = self.dcb.copy()
        self.hidden.restore()

    def store(self):
        """backup the parameters of the whole model.
        """
        self.Cb = self.C.copy()
        self.Vb = self.V.copy()
        self.Vcb = self.Vc.copy()
        if self.en_direct:
            self.Mb = self.M.copy()
            self.Mcb = self.Mc.copy()
        if self.en_bias:
            self.db = self.d.copy()
            self.dcb = self.dc.copy()
        self.hidden.store()

    def adjust(self, entropy):
        """Adjust the learning rate according to the entropy on validation data set.
        :Param entropy: the current entropy on validation data set.
        """
        if entropy > self.pre_entropy:
            self.restore()
        else:
            self.store()
        if entropy * self.min_improve > self.pre_entropy:
            if self.adjust_alpha: return True
            self.adjust_alpha = True
        if self.adjust_alpha:
            self.alpha = self.alpha * self.alpha_cut
        self.pre_entropy = entropy
        return False

    def run(self, sentence):
        """Run language model.
        """
        self.words = [self.vocab.get_word(self.sentence_start),]
        if self.input_unit == 'W':
            self.words.extend([self.vocab.get_word(word) for word in sentence.split()])
        else:
            self.words.extend([self.vocab.get_word(word) for word in sentence])
        self.words.append(self.vocab.get_word(self.sentence_end))
        T = len(self.words) - 1
        if self.is_fnn:
            vectors =  np.array([self.C[self.words[0].index], ] * (self.gram_order-2)
                + [self.C[word.index] for word in self.words])
            self.x = np.zeros((T, self.input_size))
            for t in xrange(T):
                self.x[t] = vectors[t:t+self.gram_order-1].ravel()
        else:
            self.x = [self.C[word.index] for word in self.words[:-1]]
        self.s = self.hidden.run(self.x)
        self.y = np.zeros((T, self.vocab_size))
        self.yc = np.zeros((T, self.class_size))
        for t in xrange(T):
            word = self.words[t+1]
            start, end = self.vocab.get_range(word.cindex)
            self.y[t][start:end+1] = np.dot(self.V[start:end+1], self.s[t])
            self.yc[t] = np.dot(self.Vc, self.s[t])
            if self.en_direct:
                self.y[t][start:end+1] += np.dot(self.M[start:end+1], self.x[t])
                self.yc[t] += np.dot(self.Mc, self.x[t])
            if self.en_bias:
                self.y[t][start:end+1] += self.d[start:end+1]
                self.yc[t] += self.dc
            self.y[t][start:end+1] = self.softmax(np.clip(self.y[t][start:end+1], -50, 50))
            self.yc[t] = self.softmax(np.clip(self.yc[t], -50, 50))
            if word.name == self.unknown_word: continue
            self.word_num += 1
            self.logp += np.log(self.y[t][word.index] * self.yc[t][word.cindex])

    def reshape(self, dLdx):
        """Reshape dLdx when the first neural layer is feedforward one.
        :Param dLdx: error gradient for input feature vector 
        """
        T = len(self.words) - 1
        re_dLdx = np.zeros((T, self.vector_dim))
        for t in xrange(T):
            re_dLdxt = dLdx[t].reshape((self.gram_order-1, self.vector_dim))
            for i in xrange(self.gram_order-1):
                re_dLdx[max(0, t-self.gram_order+i+2)] += re_dLdxt[i]
        return re_dLdx

    def update(self):
        """Update parameters of this language model during training.
        """
        T = len(self.words) - 1
        dLdx = np.zeros((T, self.input_size))
        dLds = np.zeros((T, self.hidden_size))
        for t in xrange(T):
            word = self.words[t+1]
            dLdy = 0 - self.y[t]
            dLdc = 0 - self.yc[t]
            dLdy[word.index] += 1
            dLdc[word.cindex] += 1
            start, end = self.vocab.get_range(word.cindex)
            dLdp = dLdy[start:end+1]
            if self.en_direct:
                dLdx[t] += np.dot(self.M[start:end+1].T, dLdp)
                dLdM = np.clip(np.outer(dLdp, self.x[t]), -15, 15)
                self.M[start:end+1] += self.alpha * dLdM - self.beta * self.M[start:end+1]
                dLdx[t] += np.dot(self.Mc.T, dLdc)
                dLdMc = np.clip(np.outer(dLdc, self.x[t]), -15, 15)
                self.Mc += self.alpha * dLdMc - self.beta * self.Mc
            if self.en_bias:
                self.d[start:end+1] += self.alpha * np.clip(dLdp, -15, 15) - self.beta * self.d[start:end+1]
                self.dc += self.alpha * np.clip(dLdc, -15, 15) - self.beta * self.dc
            dLds[t] += np.dot(self.V[start:end+1].T, dLdp)
            dLds[t] += np.dot(self.Vc.T, dLdc)
            self.V[start:end+1] += self.alpha * np.clip(np.outer(dLdp, self.s[t]), -15, 15) - self.beta * self.V[start:end+1]
            self.Vc += self.alpha * np.clip(np.outer(dLdc, self.s[t]), -15, 15) - self.beta * self.Vc
        dLdx += self.hidden.update(dLds, self.alpha, self.beta)
        if self.is_fnn: dLdx = self.reshape(dLdx)
        for t in xrange(T):
            index = self.words[t].index
            self.C[index] += np.clip(dLdx[t], -15, 15) * self.alpha - self.C[index] * self.beta

    def train(self):
        """Train this language model.
        """
        output = codecs.open(self.output_file+'.txt', 'a', encoding='utf-8')
        output.write('Training Time: %s\n' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        file_num = len(os.listdir(self.train_files))
        # for each iteration
        for i in xrange(self.iter_num):
            self.logp = 0  # sum of the logarithm probability
            self.word_num = 0  # count the number of words
            begin = time.clock()  # record begin time of each iteration
            # for each training file
            for findex, file_name in enumerate(glob(os.path.join(self.train_files, '*'))):
                input_file = codecs.open(file_name, self.file_mode, encoding='utf-8')
                sentences =  input_file.readlines()
                sentence_num = len(sentences)
                for sindex, sentence in enumerate(sentences):
                    print 'NNLM: %sth iteration, %d/%d files, %d/%d sentences.\r' % (
                        ('%d' % (i+1)).zfill(2), (findex+1), file_num, sindex, sentence_num),
                    self.run(sentence.strip())
                    self.update()
                input_file.close()
            log = 'NNLM: %sth iteration, %d files, elapsed time is %.2fs, training entropy is %.2f, alpha is %.5f.' % (
                ('%d' % (i+1)).zfill(2), file_num, (time.clock()-begin), (-self.logp/np.log(2)/float(self.word_num)), self.alpha)
            print log
            output.write(log+'\n')
            # run over validation data set
            if self.adjust(self.valid(output)): break
            # backup the parameters of whole model
        output.close()

    def valid(self, output):
        """Adjust learning rate using validation data set.
        """
        self.logp = 0  # the logarithm probability of test data set
        self.word_num = 0  # total number of words in test data set
        begin = time.clock()  # record begin time of validation
        file_num = len(os.listdir(self.valid_files))
        for findex, file_name in enumerate(glob(os.path.join(self.valid_files, '*'))):
            input_file = codecs.open(file_name, self.file_mode, encoding='utf-8')
            sentences = input_file.readlines()
            sentence_num = len(sentences)
            for sindex, sentence in enumerate(sentences):
                print 'NNLM: Validation, %d/%d files, %d/%d sentences.\r' % (
                    (findex+1), file_num, sindex, sentence_num),
                self.run(sentence.strip())
            input_file.close()
        entropy = -self.logp/np.log(2)/float(self.word_num)
        log = 'NNLM: Validation, %d files, elapsed time is %.2fs, validation entropy is %.2f.' % (
            file_num, (time.clock()-begin), entropy)
        print log
        output.write(log+'\n')
        return entropy

    def test(self):
        """Test this language model.
        """
        self.logp = 0  # the logarithm probability of test data set
        self.word_num = 0  # total number of words in test data set
        begin = time.clock()  # record begin time of test
        file_num = len(os.listdir(self.test_files))
        output = codecs.open(self.output_file+'.txt', 'a', encoding='utf-8')
        output.write('Training Time: %s\n' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # for each test file
        for findex, file_name in  enumerate(glob(os.path.join(self.test_files, '*'))):
            input_file = codecs.open(file_name, self.file_mode, encoding='utf-8')
            sentences =  input_file.readlines()
            sentence_num = len(sentences)
            # for each sentence
            for sindex, sentence in enumerate(sentences):
                print 'NNLM: Testing, %d/%d files, %d/%d sentences.\r' % (
                    (findex+1), file_num, sindex, sentence_num),
                self.run(sentence.strip())
            input_file.close()
        ppl = np.exp(-self.logp/float(self.word_num))
        log = 'NNLM: Test, %d files, elapsed time is %.2fs, PPL is %.2f.' % ( 
                file_num, (time.clock()-begin), ppl)
        print log
        output.write(log+'\n')
        output.close()

    def save(self):
        """Save the whole language model.
        """
        pickle.dump(self, open(self.output_file, 'w'))