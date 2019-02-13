print('HAN')
import matplotlib.pyplot as plt
import os,re
import numpy as np
import logging
import sys
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split, RandomizedSearchCV,StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import make_scorer,f1_score,recall_score,fbeta_score,\
precision_recall_fscore_support,accuracy_score,precision_score
#from lr_finder import LRFinder
from keras.callbacks import LearningRateScheduler
##############################################################################
#layers.py

import keras
from keras import backend as K

class AttentionLayer(keras.layers.Layer):
    def __init__(self, context_vector_length=100, **kwargs):
        """
        An implementation of a attention layer. This layer
        accepts a 3d Tensor (batch_size, time_steps, input_dim) and
        applies a single layer attention mechanism in the time
        direction (the second axis).
        :param context_vector_lenght: (int) The size of the hidden context vector.
            If set to 1 this layer reduces to a standard attention layer.
        :param kwargs: Any argument that the baseclass Layer accepts.
        """
        self.context_vector_length = context_vector_length
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[2]

        # Add a weights layer for the
        self.W = self.add_weight(
            name='W', shape=(dim, self.context_vector_length),
            initializer=keras.initializers.get('uniform'),
            trainable=True
        )

        self.u = self.add_weight(
            name='context_vector', shape=(self.context_vector_length, 1),
            initializer=keras.initializers.get('uniform'),
            trainable=True
        )

        super(AttentionLayer, self).build(input_shape)

    def _get_attention_weights(self, X):
        """
        Computes the attention weights for each timestep in X
        :param X: 3d-tensor (batch_size, time_steps, input_dim)
        :return: 2d-tensor (batch_size, time_steps) of attention weights
        """
        # Compute a time-wise stimulus, i.e. a stimulus for each
        # time step. For this first compute a hidden layer of
        # dimension self.context_vector_length and take the
        # similarity of this layer with self.u as the stimulus
        u_tw = K.tanh(K.dot(X, self.W))
        tw_stimulus = K.dot(u_tw, self.u)

        # Remove the last axis an apply softmax to the stimulus to
        # get a probability.
        tw_stimulus = K.reshape(tw_stimulus, (-1, tw_stimulus.shape[1]))
        att_weights = K.softmax(tw_stimulus)

        return att_weights

    def call(self, X):
        att_weights = self._get_attention_weights(X)

        # Reshape the attention weights to match the dimensions of X
        att_weights = K.reshape(att_weights, (-1, att_weights.shape[1], 1))
        att_weights = K.repeat_elements(att_weights, X.shape[-1], -1)

        # Multiply each input by its attention weights
        weighted_input = keras.layers.Multiply()([X, att_weights])

        # Sum in the direction of the time-axis.
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def get_config(self):
        config = {
            'context_vector_length': self.context_vector_length
        }
        base_config = super(AttentionLayer, self).get_config()
        return {**base_config, **config}


##############################################################################
#model.py
        
from keras.layers import (
    Dense, GRU, TimeDistributed, Input,
    Embedding, Bidirectional, Lambda
)
from keras.models import Model
#from keras_han.layers import AttentionLayer


class HAN(Model):
    def __init__(
            self, max_words, max_sentences, output_size,
            embedding_matrix, word_encoding_dim,
            sentence_encoding_dim, 
            kernel_initializer,
                      bias_initializer,
                      kernel_regularizer,
                      recurrent_regularizer,
                      bias_regularizer,
                      activity_regularizer,
                      kernel_constraint,
                      recurrent_constraint,
                      bias_constraint,
                      dropout,
                      recurrent_dropout,
                      kernel_initializer2,
                      bias_initializer2,
                      kernel_regularizer2,
                      recurrent_regularizer2,
                      bias_regularizer2,
                      activity_regularizer2,
                      kernel_constraint2,
                      recurrent_constraint2,
                      bias_constraint2,
                      dropout2,
                      recurrent_dropout2,
                      inputs=None,
            outputs=None, name='han-for-docla'
    ):
        """
        A Keras implementation of Hierarchical Attention networks
        for document classification.
        :param max_words: The maximum number of words per sentence
        :param max_sentences: The maximum number of sentences
        :param output_size: The dimension of the last layer (i.e.
            the number of classes you wish to predict)
        :param embedding_matrix: The embedding matrix to use for
            representing words
        :param word_encoding_dim: The dimension of the GRU
            layer in the word encoder.
        :param sentence_encoding_dim: The dimension of the GRU
            layer in the sentence encoder.
        """
        self.max_words = max_words
        self.max_sentences = max_sentences
        self.output_size = output_size
        self.embedding_matrix = embedding_matrix
        self.word_encoding_dim = word_encoding_dim
        self.sentence_encoding_dim = sentence_encoding_dim
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.recurrent_constraint = recurrent_constraint
        self.bias_constraint = bias_constraint
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.kernel_initializer2 = kernel_initializer2
        self.bias_initializer2 = bias_initializer2
        self.kernel_regularizer2 = kernel_regularizer2
        self.recurrent_regularizer2 = recurrent_regularizer2
        self.bias_regularizer2 = bias_regularizer2
        self.activity_regularizer2 = activity_regularizer2
        self.kernel_constraint2 = kernel_constraint2
        self.recurrent_constraint2 = recurrent_constraint2
        self.bias_constraint2 = bias_constraint2
        self.dropout2 = dropout2
        self.recurrent_dropout2 = recurrent_dropout2
        
        
        in_tensor, out_tensor = self._build_network()

        super(HAN, self).__init__(
            inputs=in_tensor, outputs=out_tensor, name=name
        )

    def build_word_encoder(self, max_words, embedding_matrix,encoding_dim=200):
        """
        Build the model that embeds and encodes in context the
        words used in a sentence. The return model takes a tensor of shape
        (batch_size, max_length) that represents a collection of sentences
        and returns an encoded representation of these sentences.
        :param max_words: (int) The maximum sentence length this model accepts
        :param embedding_matrix: (2d array-like) A matrix with the i-th row
            representing the embedding of the word represented by index i.
        :param encoding_dim: (int, should be even) The dimension of the
            bidirectional encoding layer. Half of the nodes are used in the
            forward direction and half in the backward direction.
        :return: Instance of keras.Model
        """
        assert encoding_dim % 2 == 0, "Embedding dimension should be even"

        vocabulary_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        embedding_layer = Embedding(
            vocabulary_size, embedding_dim,
            weights=[embedding_matrix], input_length=max_words,
            trainable=False
        )

        sentence_input = Input(shape=(max_words,), dtype='int32')
        embedded_sentences = embedding_layer(sentence_input)
        encoded_sentences = Bidirectional(
            GRU(int(encoding_dim / 2), return_sequences=True,
                      kernel_initializer = self.kernel_initializer,
                      bias_initializer = self.bias_initializer,
                      kernel_regularizer = self.kernel_regularizer,
                      recurrent_regularizer = self.recurrent_regularizer,
                      bias_regularizer = self.bias_regularizer,
                      activity_regularizer = self.activity_regularizer,
                      kernel_constraint = self.kernel_constraint,
                      recurrent_constraint = self.recurrent_constraint,
                      bias_constraint = self.bias_constraint,
                      dropout = self.dropout,
                      recurrent_dropout = self.recurrent_dropout
                )
        )(embedded_sentences)

        return Model(
            inputs=[sentence_input], outputs=[encoded_sentences], name='word_encoder'
        )

    def build_sentence_encoder(self, max_sentences, summary_dim, encoding_dim):
        """
        Build the encoder that encodes the vector representation of
        sentences in their context.
        :param max_sentences: The maximum number of sentences that can be
            passed. Use zero-padding to supply shorter sentences.
        :param summary_dim: (int) The dimension of the vectors that summarizes
            sentences. Should be equal to the encoding_dim of the word
            encoder.
        :param encoding_dim: (int, even) The dimension of the vector that
            summarizes sentences in context. Half is used in forward direction,
            half in backward direction.
        :return: Instance of keras.Model
        """
        assert encoding_dim % 2 == 0, "Embedding dimension should be even"

        text_input = Input(shape=(max_sentences, summary_dim))
        encoded_sentences = Bidirectional(
            GRU(int(encoding_dim / 2), return_sequences=True,
                kernel_initializer = self.kernel_initializer2,
                      bias_initializer = self.bias_initializer2,
                      kernel_regularizer = self.kernel_regularizer2,
                      recurrent_regularizer = self.recurrent_regularizer2,
                      bias_regularizer = self.bias_regularizer2,
                      activity_regularizer = self.activity_regularizer2,
                      kernel_constraint = self.kernel_constraint2,
                      recurrent_constraint = self.recurrent_constraint2,
                      bias_constraint = self.bias_constraint2,
                      dropout = self.dropout2,
                      recurrent_dropout = self.recurrent_dropout2)
        )(text_input)
        return Model(
            inputs=[text_input], outputs=[encoded_sentences], name='sentence_encoder'
        )

    def _build_network(self):
        """
        Build the graph that represents this network
        :return: in_tensor, out_tensor, Tensors representing the input and output
            of this network.
        """
        in_tensor = Input(shape=(self.max_sentences, self.max_words))

        word_encoder = self.build_word_encoder(
            self.max_words, self.embedding_matrix, self.word_encoding_dim
        )

        word_rep = TimeDistributed(
            word_encoder, name='word_encoder'
        )(in_tensor)

        # Sentence Rep is a 3d-tensor (batch_size, max_sentences, word_encoding_dim)
        sentence_rep = TimeDistributed(
            AttentionLayer(), name='word_attention'
        )(word_rep)

        doc_rep = self.build_sentence_encoder(
            self.max_sentences, self.word_encoding_dim, self.sentence_encoding_dim
        )(sentence_rep)

        # We get the final representation by applying our attention mechanism
        # to the encoded sentences
        doc_summary = AttentionLayer(name='sentence_attention')(doc_rep)
        
        out_tensor = Dense(
            self.output_size, activation='sigmoid', name='class_prediction' #softmax for categories
        )(doc_summary)

        return in_tensor, out_tensor

    def get_config(self):
        config = {
            'max_words': self.max_words,
            'max_sentences': self.max_sentences,
            'output_size': self.output_size,
            'embedding_matrix': self.embedding_matrix,
            'word_encoding_dim': self.word_encoding_dim,
            'sentence_encoding_dim': self.sentence_encoding_dim,
            'base_config': super(HAN, self).get_config()
        }

        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """
        Keras' API isn't really extendible at this point
        therefore we need to use a bit hacky solution to
        be able to correctly reconstruct the HAN model
        from a config. This therefore does not reconstruct
        a instance of HAN model, but actually a standard
        Keras model that behaves exactly the same.
        """
        base_config = config.pop('base_config')

        return Model.from_config(
            base_config, custom_objects=custom_objects
        )

    def predict_sentence_attention(self, X):
        """
        For a given set of texts predict the attention
        weights for each sentence.
        :param X: 3d-tensor, similar to the input for predict
        :return: 2d array (num_obs, max_sentences) containing
            the attention weights for each sentence
        """
        att_layer = self.get_layer('sentence_attention')
        prev_tensor = att_layer.input

        # Create a temporary dummy layer to hold the
        # attention weights tensor
        dummy_layer = Lambda(
            lambda x: att_layer._get_attention_weights(x)
        )(prev_tensor)

        return Model(self.input, dummy_layer).predict(X)



##############################################################################

# Create a logger to provide info on the state of the
# script
stdout = logging.StreamHandler(sys.stdout)
stdout.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger = logging.getLogger('default')
logger.setLevel(logging.INFO)
logger.addHandler(stdout)

MAX_WORDS_PER_SENT = 100 #TUNE?
MAX_SENT = 15 #TUNE?
max_words = 1000 #raise to 10000
embedding_dim = 300
TEST_SPLIT = 0.2

#####################################################
# Pre processing                                    #
#####################################################
logger.info("Pre-processsing data.")

# Load Kaggle's IMDB example data
data = pd.read_csv('labeledTrainData.tsv', sep='\t')


# Do some basic cleaning of the review text
def remove_quotations(text):
    """
    Remove quotations and slashes from the dataset.
    """
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    return text


def remove_html(text):
    """
    Very, very raw parser to remove HTML tags from
    texts.
    """
    tags_regex = re.compile(r'<.*?>')
    return tags_regex.sub('', text)


data['review'] = data['review'].apply(remove_quotations)
data['review'] = data['review'].apply(remove_html)
data['review'] = data['review'].apply(lambda x: x.strip().lower())

# Get the data and the sentiment
reviews = data['review'].values
target = data['sentiment'].values
del data


#####################################################
# Tokenization                                      #
#####################################################
logger.info("Tokenization.")

# Build a Keras Tokenizer that can encode every token
word_tokenizer = Tokenizer(num_words=max_words)
word_tokenizer.fit_on_texts(reviews)

# Construct the input matrix. This should be a nd-array of
# shape (n_samples, MAX_SENT, MAX_WORDS_PER_SENT).
# We zero-pad this matrix (this does not influence
# any predictions due to the attention mechanism.
X = np.zeros((len(reviews), MAX_SENT, MAX_WORDS_PER_SENT), dtype='int32')

for i, review in enumerate(reviews):
    sentences = sent_tokenize(review)
    tokenized_sentences = word_tokenizer.texts_to_sequences(
        sentences
    )
    tokenized_sentences = pad_sequences(
        tokenized_sentences, maxlen=MAX_WORDS_PER_SENT
    )

    pad_size = MAX_SENT - tokenized_sentences.shape[0]

    if pad_size < 0:
        tokenized_sentences = tokenized_sentences[0:MAX_SENT]
    else:
        tokenized_sentences = np.pad(
            tokenized_sentences, ((0,pad_size),(0,0)),
            mode='constant', constant_values=0
        )

    # Store this observation as the i-th observation in
    # the data matrix
    X[i] = tokenized_sentences[None, ...]

# Transform the labels into a format Keras can handle
y = np.asarray(target)#to_categorical(labels)

# We make a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)
X = X_train[:500]
Y = y_train[:500]
x_test = X_test[:100]
y_test = y_test[:100]
#####################################################
# Word Embeddings                                   #
#####################################################
logger.info(
    "Creating embedding matrix using pre-trained w2v vectors."
)

# Now, we need to build the embedding matrix. For this we use
# a pretrained (on the wikipedia corpus) 100-dimensional GloVe
# model.

# Load the embeddings from a file
embeddings = {}
#r'C:\Users\Enrico\Desktop\Projet Innovation\
with open('imdb_w2v.txt', encoding='utf-8') as file:
    dummy = file.readline()
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')

        embeddings[word] = coefs

# Initialize a matrix to hold the word embeddings
embedding_matrix = np.random.random(
    (len(word_tokenizer.word_index) + 1, embedding_dim)
)

# Let the padded indices map to zero-vectors. This will
# prevent the padding from influencing the results
embedding_matrix[0] = 0

# Loop though all the words in the word_index and where possible
# replace the random initalization with the GloVe vector.
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


#####################################################
# Model Training                                    #
#####################################################
def create_model(optimizer,
                 kernel_initializer,
                     bias_initializer,
                      kernel_regularizer,
                      recurrent_regularizer,
                      bias_regularizer,
                      activity_regularizer,
                      kernel_constraint,
                      recurrent_constraint,
                      bias_constraint,
                      dropout,
                      recurrent_dropout,
                      kernel_initializer2,
                     bias_initializer2,
                      kernel_regularizer2,
                      recurrent_regularizer2,
                      bias_regularizer2,
                      activity_regularizer2,
                      kernel_constraint2,
                      recurrent_constraint2,
                      bias_constraint2,
                      dropout2,
                      recurrent_dropout2,
                      word_encoding_dim,
                      sentence_encoding_dim
                      ):


    han_model = HAN(
        MAX_WORDS_PER_SENT, MAX_SENT, 1, embedding_matrix, #2 is output size
        word_encoding_dim, sentence_encoding_dim, #number of units fir the 2 GRUs
                      kernel_initializer,
                      bias_initializer,
                      kernel_regularizer,
                      recurrent_regularizer,
                      bias_regularizer,
                      activity_regularizer,
                      kernel_constraint,
                      recurrent_constraint,
                      bias_constraint,
                      dropout,
                      recurrent_dropout,
                      kernel_initializer2,
                      bias_initializer2,
                      kernel_regularizer2,
                      recurrent_regularizer2,
                      bias_regularizer2,
                      activity_regularizer2,
                      kernel_constraint2,
                      recurrent_constraint2,
                      bias_constraint2,
                      dropout2,
                      recurrent_dropout2
    )
    
    han_model.summary()
    
    han_model.compile(optimizer=optimizer, loss='binary_crossentropy')
    
    #attentionWeights = han_model.predict_sentence_attention(X)

    #np.savetxt("attention.csv", attentionWeights, delimiter=",", fmt='%s', header=None)
    
    return han_model

f=open('HAN-gridoutput.txt','w')
for i in range(1): #No need for a double cross validation
    #epochs = [int((word_encoding_dim+sentence_encoding_dim)/32+np.log2(word_encoding_dim+sentence_encoding_dim))]
    batch_size = [16,32,64,128]
    epochs = [50]
    word_encoding_dim = [64,100,128,200,256]
    sentence_encoding_dim = [16,32,64,100,128]
    
    #GRU params
    #activation = ['hard_sigmoid','softmax','elu','selu','softplus','softsign','relu',
    #                      'tanh','sigmoid','exponential','linear','PReLU','LeakyReLu']
    #recurrent_activation=['hard_sigmoid','softmax','elu','selu','softplus','softsign','relu',
    #                      'tanh','sigmoid','exponential','linear','PReLU','LeakyReLu']
    kernel_initializer=['glorot_normal','glorot_uniform','TruncatedNormal','VarianceScaling'] #cause it's a tanh
    #                    'zeros','ones','constant','RandomNormal','RandomUniform',
    #                  'TruncatedNormal','VarianceScaling','orthogonal','identity',
    #
    #                  'he_uniform','he_normal']
    #recurrent_initializer=['zeros',
    #                       'ones','constant','RandomNormal','RandomUniform',
    #                  'TruncatedNormal','VarianceScaling','orthogonal','identity',
    #                  'lecun_uniform','lecun_normal','glorot_uniform','glorot_normal',
    #                  'he_uniform','he_normal']
    bias_initializer=['zeros','ones','glorot_normal','he_normal']
    #                   'ones','constant','RandomNormal','RandomUniform',
    #                  'TruncatedNormal','VarianceScaling','orthogonal','identity',
    #                  'lecun_uniform','lecun_normal','glorot_uniform','glorot_normal',
    #                  'he_uniform','he_normal']
    kernel_regularizer=[None, 'l1','l2','l1_l2']
    recurrent_regularizer=[None, 'l1','l2','l1_l2']
    bias_regularizer=[None, 'l1','l2','l1_l2']
    activity_regularizer=[None, 'l1','l2','l1_l2']
    kernel_constraint=[None, 'MaxNorm']#'MinMaxNorm','NonNeg','UnitNorm',
    recurrent_constraint=[None, 'MaxNorm']#'MinMaxNorm','NonNeg','UnitNorm',
    bias_constraint=[None, 'MaxNorm']#'MinMaxNorm','NonNeg','UnitNorm',
    dropout=[0.0, 0.2,0.3,0.4,0.5]
    recurrent_dropout=[0.0, 0.2,0.3,0.4,0.5]
    
    optimizer = ['Adadelta','Adam','Adamax','Nadam']
    #Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,clipvalue,clipnorm)
    
    param_grid = dict(word_encoding_dim = word_encoding_dim,
                      sentence_encoding_dim = sentence_encoding_dim,
                      optimizer = optimizer,
                      #activation = activation,
                      #recurrent_activation = recurrent_activation,
                     kernel_initializer = kernel_initializer,
                      #recurrent_initializer =  recurrent_initializer,
                      bias_initializer = bias_initializer,
                      kernel_regularizer = kernel_regularizer,
                      recurrent_regularizer = recurrent_regularizer,
                      bias_regularizer = bias_regularizer,
                      activity_regularizer = activity_regularizer,
                      kernel_constraint = kernel_constraint,
                      recurrent_constraint = recurrent_constraint,
                      bias_constraint = bias_constraint,
                      dropout = dropout,
                      recurrent_dropout = recurrent_dropout,
                     kernel_initializer2 = kernel_initializer,
                      bias_initializer2 = bias_initializer,
                      kernel_regularizer2 = kernel_regularizer,
                      recurrent_regularizer2 = recurrent_regularizer,
                      bias_regularizer2 = bias_regularizer,
                      activity_regularizer2 = activity_regularizer,
                      kernel_constraint2 = kernel_constraint,
                      recurrent_constraint2 = recurrent_constraint,
                      bias_constraint2 = bias_constraint,
                      dropout2 = dropout,
                      recurrent_dropout2 = recurrent_dropout,
                      batch_size = batch_size,
                      epochs = epochs
                      )
    
    scoring = {'acc':make_scorer(accuracy_score),'f1': make_scorer(f1_score),'f2': make_scorer(fbeta_score, beta=2),
               'rec': make_scorer(recall_score)}
    
    model = KerasClassifier(build_fn=create_model,verbose=1 )
    
    grid = RandomizedSearchCV(cv=2,n_iter=2,
                              estimator=model, param_distributions=param_grid,
                              n_jobs=-1,
                              scoring=scoring,
                              refit='acc', #or f1, f2
                              return_train_score = True
                              #random_state = 42
                              )
    
    
    """#fix
    trainingDim = 500*3/4 
    lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-1, 
                                 steps_per_epoch=np.ceil(trainingDim/batch_size[1]), 
                                 epochs=3)
    """
    def step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=10):
        '''
        Wrapper function to create a LearningRateScheduler with step decay schedule.
        '''
        def schedule(epoch):
            return initial_lr * (decay_factor ** np.floor(epoch/step_size))
        
        return LearningRateScheduler(schedule)

    lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)    
                       
    grid_result = grid.fit(X, Y, callbacks=[lr_sched])    
    
    """   
    lr_finder.plot_loss('lr_loss.png')
    lr_finder.plot_lr('lr.png')
    """
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
    mean_acc = grid_result.cv_results_['mean_test_acc']
    mean_f1 = grid_result.cv_results_['mean_test_f1']
    mean_f2 = grid_result.cv_results_['mean_test_f2']
    mean_rec = grid_result.cv_results_['mean_test_rec']
    params = grid_result.cv_results_['params']
    #look into grid_result_v: rank_test_rec, rank_test_f2 give a ranking of the models for both parameters++
    
    for mean0, mean1, mean2, mean3, param in zip(mean_acc, mean_f1, mean_f2, mean_rec, params):
        f.write("acc %f f1 %f f2 %f rec %f with: %r\n" % (mean0, mean1, mean2, mean3, param))
    f.write('---------------------------------------------------------------\n')

res = pd.DataFrame(grid.cv_results_)
res.to_csv('HAN_params.csv')

#Test here
y_pred = grid.predict(x_test)
f.write("The final accuracy is: ")
somme = 0
for i in range(len(y_test)):
    if y_test[i]==y_pred[i]:
        somme+=1
print(somme,len(y_test))
avg = somme/len(y_test)
f.write("%f"%avg)
f.close()

print('y_pred:',y_pred)
print('y_test',y_test)




"""
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('HAN_acc.png')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('HAN_loss.png')

f=open('out_HAN.txt','w')
temp=''
for i in range(len(acc)):
    temp+=str('At epoch:',i,'.....Accuracy is:',acc[i],'Validation accuracy is:',val_acc[i])
    temp+='\n'
f.write(temp)
f.close()
"""
