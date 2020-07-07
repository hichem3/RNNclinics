################################
# Authors : Enrico Sartor, Loic Verlingue
################################

"""
Functions to build a han_model containing custom Keras layers that use the attention mechanism.
This function can be used to run new training or to build a han_model to load trained weights (instead of loading custom objects in saved model).
If runing new training, setting the weights of the embedding layer might be a good idea
"""
'''
################
# directories
################
out_file = 'HAN_100epoch10eval'
data_dir="/mnt/beegfs/scratch/l_verlingue/NLP/CondaCloneWrkFlow/data/"
results_dir="/mnt/beegfs/scratch/l_verlingue/NLP/CondaCloneWrkFlow/results/"

#################

# hyperparameters
#################
MAX_WORDS_PER_SENT = 40
MAX_SENT = 80
max_words = 10000
embedding_dim = 128
TEST_SPLIT = 0.2
word_encoding_dim=256
sentence_encoding_dim=256
l1=0
l2=0
dropout=0.2
MAX_EVALS = 10 # number of models to evaluate with hyperopt
Nepochs = 100
'''

#import os
#import pandas as pd
import numpy as np
import keras
from keras import backend as K
from keras.layers import (
    Dense, GRU, TimeDistributed, Input,
    Embedding, Bidirectional, Lambda, Dropout
)
from keras.models import Model
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras import regularizers


#from keras_han.layers import AttentionLayer

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


class HAN(Model):
    def __init__(
            self, max_words, max_sentences, output_size,
            embedding_matrix, word_encoding_dim=200,
            sentence_encoding_dim=200, 
            l1=0.,l2=0.,dropout=0.,
            inputs=None,outputs=None, name='han-for-docla'
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
        self.l1 = l1
        self.l2 = l2
        self.dropout = dropout

        in_tensor, out_tensor = self._build_network()

        super(HAN, self).__init__(
            inputs=in_tensor, outputs=out_tensor, name=name
        )

    def build_word_encoder(self, max_words, embedding_matrix, encoding_dim=200):
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
            GRU(int(encoding_dim / 2), return_sequences=True)
        )(embedded_sentences)

        return Model(
            inputs=[sentence_input], outputs=[encoded_sentences], name='word_encoder'
        )

    def build_sentence_encoder(self, max_sentences, summary_dim, encoding_dim=200):
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
            GRU(int(encoding_dim / 2), return_sequences=True)
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
        doc_summary0 = AttentionLayer(name='sentence_attention')(doc_rep)
        
        doc_summary = Dropout(self.dropout)(doc_summary0)
        
        out_tensor = Dense(
            self.output_size, activation='sigmoid', name='class_prediction',
            kernel_regularizer = keras.regularizers.l1_l2(l1=self.l1, l2=self.l2)
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

'''
##################
# embedding matrix
##################

# Initialize a matrix to hold the word embeddings
embedding_matrix = np.random.random(
    (max_words + 1, embedding_dim)
)


##################
#create model
##################

han_model = HAN(
        MAX_WORDS_PER_SENT, MAX_SENT, 1, embedding_matrix,  # 1 is output size
        word_encoding_dim,
        sentence_encoding_dim,
        l1,  
        l2,  
        dropout)
'''