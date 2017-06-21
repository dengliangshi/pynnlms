#encoding=utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Libraries
import os
import codecs
from glob import glob

# Third-party Libraries
import numpy as np

# User Define Modules
from word import Word

# --------------------------------------------------------Global Strings----------------------------------------------------

# ----------------------------------------------------------Class Main------------------------------------------------------
class Vocab(object):
    """Buiding up vocabulary from given data set.
    """
    def __init__(self):
        """Initialization function.
        """
        # size of vocabulary
        self.vocab_size = 0
        # number of words learned from training data
        self.word_count = 0
        # words and their frequency like {word: freq, ...}
        self.freq = {}
        # word in vocabulary, like {word: Word, ...}
        self.vocab = {}
        # word index for each class, like: {class: (start_index, end_index)}
        self.crange = {}

    def __repr__(self):
        """Instance display format.
        """
        return '<Vocabulary: %d words>' % self.vocab_size

    def init_model(self, config):
        """Initialize vocabulary.
        :Param config: configuration of language model
        """
        # size of vocabulary
        self.vocab_size = config.get('VOCAB_SIZE')
        # directory of training files
        self.data_files = config.get('TRAIN_FILES')
        # the format of training data
        self.file_type = config.get('FILE_TYPE')
        # the size of feature vector
        self.input_unit = config.get('INPUT_UNIT')
        # the number of words' classes
        self.word_class = config.get('CLASS_SIZE')
        # the flag for the start of sentence
        self.sentence_start = config.get('SENTENCE_START')
        # the flag for the end of sentence
        self.sentence_end = config.get('SENTENCE_END')
        # out of vocabulary
        self.unknown_word = config.get('UNKNOWN_WORD')
        # add start mark and end mark of sentence 
        self.freq[self.sentence_start] = 0
        self.freq[self.sentence_end] = 0
        self.word_count = 2

    def add_words(self, input_file, file_mode):
        """Add words into vocabulary form a file
        :Param input_file: data file.
        :Param file_mode: the mode in which data file will be opened
        """
        sentences = codecs.open(input_file, file_mode, encoding='utf-8')
        for sentence in sentences:
            if not sentence.strip(): continue
            self.freq[self.sentence_start] += 1
            self.freq[self.sentence_end] += 1
            if self.input_unit == 'W':
                units = sentence.strip().split() 
            else:
                units = list(sentence.strip())
            for unit in units:
                if unit in self.freq:
                    self.freq[unit] += 1
                else:
                    self.freq[unit] = 1
                    self.word_count += 1
        sentences.close()

    def truncate(self):
        """Move words with low frequency out vocabulary and add their freqency to word OOV, 
        if the number of words learned from training data exceeds the specified vocabulary size.
        """
        if self.word_count < self.vocab_size -1:
            self.vocab_size = self.word_count + 1
            self.freq[self.unknown_word] = 1
        else:
            sorted_words = sorted(self.freq.iteritems(),
                key=lambda d: d[1], reverse=True)
            self.freq.clear()
            self.freq[self.unknown_word] = sum([x[1] for x in
                sorted_words[self.vocab_size-1:]])
            for word, freq in sorted_words[:self.vocab_size-1]:
                self.freq[word] = freq

    def assign(self):
        """Assign each word with feature vector and class.
        """
        index = 0
        cindex = 0
        ac_freq = 0
        start_index = 0
        sorted_words = sorted(self.freq.iteritems(),
            key=lambda d: d[1], reverse=True)
        base = sum([x[1] for x in sorted_words])
        sqrt_base = sum([np.sqrt(x[1]/float(base)) for x in sorted_words])
        for word, freq in sorted_words:
            self.vocab[word] = Word(word, index, cindex)
            if (ac_freq > (cindex+1)/float(self.word_class) and 
                cindex < self.word_class-1):
                self.crange[cindex] = (start_index, index)
                cindex += 1
                start_index = index + 1
            ac_freq += np.sqrt(freq/float(base))/sqrt_base
            index += 1
        self.crange[cindex] = (start_index, self.vocab_size-1)

    def generate(self):
        """Generate vocabulary from training data.
        """
        file_mode = 'r' if self.file_type == 'T' else 'rb'
        for data_file in glob(os.path.join(self.data_files, '*')):
            self.add_words(data_file, file_mode)
        self.truncate()
        self.assign()
        return self.vocab_size

    def get_word(self, word):
        """Get the word instance.
        :Param word: word string.
        """
        return self.vocab.get(word, self.vocab[self.unknown_word])

    def get_range(self, cindex):
        """Get the range of word index for a class.
        :Param cindex: class index.
        """
        return self.crange.get(cindex)