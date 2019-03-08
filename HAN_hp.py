print('HAN hyperopt')
out_file = 'HAN_hp.csv'
##
#
#
#Yang, Zichao, et al. "Hierarchical attention networks for document classification." 
#Proceedings of the 2016 Conference of the North American Chapter of the Association 
#for Computational Linguistics: Human Language Technologies. 2016”.
#
#Code inspired from: FlorisHoogenboom.
#https://github.com/FlorisHoogenboom/keras-han-for-docla
##

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os,re,sys,csv,logging
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import to_categorical
from nltk.tokenize import sent_tokenize
from keras import regularizers
from sklearn.model_selection import train_test_split, RandomizedSearchCV,GridSearchCV,StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import make_scorer,f1_score,recall_score,fbeta_score,\
precision_recall_fscore_support,accuracy_score,precision_score,classification_report, confusion_matrix,\
roc_curve,roc_auc_score,precision_recall_curve,average_precision_score,auc
from hyperopt import STATUS_OK,tpe,hp,Trials,fmin



def rec_scorer(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_scorer(y_true, y_pred, threshold_shift=0):
    beta = 1

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

def f2_scorer(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

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
        """
        data = np.zeros((att_weights.shape[1], 1,))
        sess = tf.Session()
        sess.run(tf.constant(att_weights))
        data = tf.constant(att_weights).eval()
        dataset = pd.DataFrame({'Column1':data[:,0],'Column2':data[:,1]})
        #array = att_weights.eval(session=get_session())
        dataset.to_csv('WEIGHTS.csv')
        """
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
                      l1,l2,
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
        self.l1 = l1
        self.l2 = l2
        
        
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
            GRU(int(encoding_dim / 2), return_sequences=True
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
            GRU(int(encoding_dim / 2), return_sequences=True
                      )
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
            self.output_size, activation='sigmoid', name='class_prediction', #softmax for categories
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

MAX_WORDS_PER_SENT = 32 #increase
MAX_SENT = 80 #increase
max_words = 10000 #increase to 10000
embedding_dim = 300
TEST_SPLIT = 0.2

#####################################################
# Pre processing                                    #
#####################################################

logger.info("Pre-processing data.")
"""
imdb_dir = 'X_REP_RAW'
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(imdb_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname),encoding='utf-8',errors='ignore')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
"""
logger.info("Pre-processing data.")

# Load Kaggle's IMDB example data
data = pd.read_csv('labeledTrainData.tsv', sep='\t')


# Do some basic cleaning of the review text
def remove_quotations(text):

    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    return text


def remove_html(text):

    tags_regex = re.compile(r'<.*?>')
    return tags_regex.sub('', text)


data['review'] = data['review'].apply(remove_quotations)
data['review'] = data['review'].apply(remove_html)
data['review'] = data['review'].apply(lambda x: x.strip().lower())

# Get the data and the sentiment
texts = data['review'].values
labels = data['sentiment'].values
del data

#####################################################
# Tokenization                                      #
#####################################################
logger.info("Tokenization.")

# Build a Keras Tokenizer that can encode every token
word_tokenizer = Tokenizer(num_words=max_words)
word_tokenizer.fit_on_texts(texts)

# Construct the input matrix. This should be a nd-array of
# shape (n_samples, MAX_SENT, MAX_WORDS_PER_SENT).
# We zero-pad this matrix (this does not influence
# any predictions due to the attention mechanism.
X = np.zeros((len(texts), MAX_SENT, MAX_WORDS_PER_SENT), dtype='int32')

for i, review in enumerate(texts):
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
y = np.asarray(labels)#to_categorical(labels)

# We make a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)

X_train=X_train[:200]
X_test=X_test[:20]
y_train=y_train[:200]
y_test=y_test[:20]

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
with open('imdb_w2v.txt', encoding='utf-8') as file:#imdb_w2v.txt . w2v_reports_128.vec
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
def create_model(params):


    han_model = HAN(
        MAX_WORDS_PER_SENT, MAX_SENT, 1, embedding_matrix, #1 is output size
        int(params['word_encoding_dim']), int(params['sentence_encoding_dim']), #number of units for the 2 GRUs
                      int(params['l1']),int(params['l2'])
    )
    
    han_model.summary()
    
    han_model.compile(optimizer=Adam(lr=int(params['lr'])), loss='binary_crossentropy', metrics=['acc',rec_scorer,f1_scorer,f2_scorer])
    
    es = EarlyStopping(monitor='val_loss', mode='min', patience = 5, verbose=1)
    mc = ModelCheckpoint('best_HAN.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    
    han_model.fit(X_train, y_train, validation_split=0.2, batch_size=int(params['batch_size']), epochs=int(params['num_epochs']),callbacks=[es, mc])
    
    scores = han_model.evaluate(X_test, y_test, verbose=0)
    
    f2 = scores[4]
    f1 = scores[3]
    rec = scores[2]
    accuracy = scores[1]
    loss = scores[0]
    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([rec,f2,f1,accuracy, loss, params])
    of_connection.close()
    
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

logger.info('Training model')
    
space = {
    'word_encoding_dim': hp.choice('word_encoding_dim', [32,64,128]),
    'sentence_encoding_dim': hp.choice('sentence_encoding_dim', [16,32,64]),
    'num_epochs':hp.quniform('num_epochs',4,5, 1), #ES, loguniform
    'batch_size': hp.choice('batch_size',[16,32,64,128]),
    'l1':hp.choice('l1',[ 0.,0.00001,0.0001,0.001,0.01]),
    'l2':hp.choice('l2',[ 0.,0.00001,0.0001,0.001,0.01]),
    'lr':hp.choice('lr',[0.00001,0.0001,0.001,0.01,0.1])
    #
}

# Trials object to track progress
bayes_trials = Trials()

# File to save first results
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['recall','f2','f1','accuracy','loss','params'])
of_connection.close()


MAX_EVALS = 3

# Optimize
best = fmin(fn = create_model, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)

print(best)



#########################################
f=open('HAN_hp_aux.csv','w')
writer = csv.writer(f)
writer.writerow(bayes_trials)
f.close()

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


logger.info('Training done')
########################################## Save the results of the grid
res = pd.DataFrame(grid.cv_results_)
res.to_csv('HAN_params.csv')

########################################## Test here
y_pred = grid.best_estimator_.predict(X_test)

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

########################################## Confusion Matrix - Test data
# Using metrics.confusion_matrix function
cm = confusion_matrix(y_test, y_pred)
data = cm.tolist()
print("cm returned from sklearn:", data)

########################################## ROC-AUC Curve
probs = grid.best_estimator_.predict_proba(X_test)
probs = probs[:, 1]
fpr, tpr, thresholds = roc_curve(y_test,probs)
AUC =  roc_auc_score(y_test, probs)
plt.figure()
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.savefig('roc_auc.png')
print('AUC=',AUC)

########################################## Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1 = f1_score(y_test, y_pred)
auc = auc(recall, precision)
ap = average_precision_score(y_test, probs)
plt.figure()
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('precision_recall.png')
print('\nf1=',f1,'auc=',auc,'ap=',ap)
"""