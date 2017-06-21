# Neural Network Language Models

This is a Neural Networl Language Models (NNLMs) toolkit which supports Feed-forward Neural Network (FNN), Recurrent Neural Network (RNN), Long Short Term Memory (LSTM) RNN, Bidirectional RNN and Bidirectional LSTM. Neural network language models with multible hidden layers also can be built with this toolkit, and the architecture of hidden layers can be different. A class-based method is adopted in this toolkit to speed up the trainin and running of neural network language model.

## Configuration
The configuration parameters for NNLM are as follows:

|            Name          |                    Description                                                        | Type  | Required | Default |
|:------------------------:|:--------------------------------------------------------------------------------------|:-----:|:--------:|:-------:|
| AC_FUN<sup>1</sup>       | Activation function for hidden layer(s)                                               | Str   |          |   tanh  |
| ALPHA                    | Learning rate                                                                         | Float |          |   0.01  |
| ALPHA_CUT                | Cutoff learning rate by this ratio when improvement is less than minimun              | Float |          |   0.75  |
| BETA                     | Regularization paramters                                                              | Float |          |   1e-6  |
| EN_BIAS                  | Enable bias terms                                                                     | Bool  |          |  False  |
| EN_DIRECT                | Enable direct connections                                                             | Bool  |          |  False  |
| FILE_TYPE                | The type of input files, supports binary and text, 'B' for binary and 'T' for text    | Str   |          |    T    |
| GATE_FUN                 | Activation function for gates in LSTM RNN                                             | Str   |          | Sigmoid |
| GRAM_ORDER               | Order of N-Gram for FNN                                                               | Int   |          |    5    |
| HIDDEN_LAYERS<sup>2</sup>| Name and size of hidden layer(s)                                                      | List  | &radic;  |         |
| INPUT_UNIT<sup>3</sup>   | Unit of input, support word or character, 'W' for word and 'C' for character          | Str   |          |    W    |
| ITERATIONS<sup>4</sup>   | Maximum number of iteration                                                           | Int   |          |    50   |
| MIN_IMPROVE<sup>4</sup>  | Minimun rate of entropy improvement on validation data                                | Float |          |   1.003 | 
| MODEL_NAME               | Specify a name for language model                                                     | Str   |          |    -    | 
| OUTPUT_PATH              | The path under which output files will be saved                                       | Str   | &radic;  |    -    |
| RANDOM_SEED              | Seed for random generator                                                             | Int   |          |    1    |
| SENTENCE_END             | Mark for the end of a sentence                                                        | Str   |          |  <\s>   |
| SENTENCE_START           | Mark for the start of a sentence                                                      | Str   |          |   <s>   |
| TEST_FILES               | The path under which test files are stored                                            | Str   | &radic;  |    -    |
| TRAIN_FILES              | The path under which training files are stored                                        | Str   | &radic;  |    -    |
| UNKNOWN_WORD             | Mark for unknown word                                                                 | Str   |          |   OOV   |
| VALID_FILES              | The path under which validation files are stored                                      | Str   | &radic;  |    -    |
| VECTOR_DIM               | Dimension of feature vector for words or characters                                   | Int   |          |    30   |
| VOCAB_SIZE<sup>5</sup>   | The size of vocabulary learned from training data                                     | Int   |          |  10000  |

*Notes:*
*1. Activation function of hidden layer(s) could be one of `tanh`, `sigmoid`, `hard_sigmoid`, `relu` and `gaussian`.*

*2. Hidden layer(s) can be one or several of `FNN`, `RNN`, `LSTM`, `BiRNN` and `BiLSTM`, and they should be given as a list of tuples containing each layer's name and its size, like [('RNN', 30), ('FNN', 20)]. Just a tuple is ok when there is only one hidden layer. The size of hidden layers should coincide with each other.*

*3. For languages, such as English, French, whose words are separated by blank character, like white space, the `INPUT_UNITE` can be set to 'W' or 'C'. Other languages, like chinese, only `INPUT_UNITE = 'C'` does work.*

*4. Training will terminate when either reaches the maximum number of iteration or that entropy improvement on validation data is less than minimum rate happens twice.*

*5. If the number of words or characters learned from training data exceeds the specified size of vocabulary, the words or characters with low frequency will not be added into vocabulary. On the opposite, the size of vocabulary wil be reset to the number of learned words.*

## Usage
The examples are given in this toolkit. More details about the languagel models built in this toolkit, please refer to [my posts](https://dengliangshi.github.io/).

## License
The module is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).