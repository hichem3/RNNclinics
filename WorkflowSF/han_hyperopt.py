##
# to check: 
# "Save the best model" line 266 : is all oK?
# the custom metrics scores in CV



# Yang, Zichao, et al. "Hierarchical attention networks for document classification."
# Proceedings of the 2016 Conference of the North American Chapter of the Association
# for Computational Linguistics: Human Language Technologies. 2016.
#
# Code inspired from: FlorisHoogenboom.
# https://github.com/FlorisHoogenboom/keras-han-for-docla
##

################
# directories
################
out_file = 'HAN_30epoch10eval'
data_dir="data/"
results_dir="results/"
scripts_dir="scripts/"
print(out_file)

#################
# hyperparameters
#################
MAX_WORDS_PER_SENT = 40
MAX_SENT = 80
max_words = 10000
embedding_dim = 128
word_encoding_dim=256
sentence_encoding_dim=256
l1=0
l2=0
dropout=0.2
MAX_EVALS = 10 # number of models to evaluate with hyperopt
Nepochs = 30

# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
import os, re, sys, csv, logging
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.layers import (
    Dense, GRU, TimeDistributed, Input,
    Embedding, Bidirectional, Lambda
)
from keras.models import Model
# from keras_han.layers import AttentionLayer

from nltk.tokenize import sent_tokenize
from keras import regularizers
from sklearn.model_selection import train_test_split, RandomizedSearchCV, \
    GridSearchCV, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, \
    classification_report, roc_auc_score, auc
    
from hyperopt import STATUS_OK, tpe, hp, Trials, fmin

import pickle # to save and load keras tokeniser

# functions to han model and custom scorers
os.chdir(scripts_dir)
from han_model import HAN
from han_model import AttentionLayer
from utils import rec_scorer, f1_scorer, f2_scorer
os.chdir(results_dir)

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


#####################################################
# Pre processing                                    #
#####################################################

logger.info("Pre-processing data.")

# Loic version

df_all=pd.read_csv(os.path.join(data_dir, "df_all.csv"), encoding='utf-8')
df_all=df_all[df_all['CompleteValues']]

labels = df_all['screenfail']
# Transform the labels into a format Keras can handle
y = np.asarray(labels)  # to_categorical(labels)

texts = df_all['value']
texts = pd.Series.tolist(texts)
split = df_all['Cohort']


#####################################################
# Tokenization                                      #
#####################################################
logger.info("Tokenization.")


# saving tokeniser
try:    
    with open(os.path.join(results_dir,out_file+'word_tokeniser'), 'rb') as handle:
        word_tokenizer = pickle.load(handle)
except FileNotFoundError:
    # Build a Keras Tokenizer that can encode every token
    word_tokenizer = Tokenizer(num_words=max_words)
    word_tokenizer.fit_on_texts(texts)    
    with open(os.path.join(results_dir,out_file+'word_tokeniser'), 'wb' ) as handle:
        pickle.dump(word_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Construct the input matrix. This should be a nd-array of
# shape (n_samples, MAX_SENT, MAX_WORDS_PER_SENT).
# We zero-pad this matrix (this does not influence
# any predictions due to the attention mechanism.
X = np.zeros((len(texts), MAX_SENT, MAX_WORDS_PER_SENT), dtype='int32')

#review=texts[3]
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
            tokenized_sentences, ((0, pad_size), (0, 0)),
            mode='constant', constant_values=0
        )

    # Store this observation as the i-th observation in
    # the data matrix
    X[i] = tokenized_sentences[None, ...]


# be sure to do the CV metrics on the Val cohort
X_train, X_val = X[df_all['Cohort']=='Train'], X[df_all['Cohort']=='Val']
y_train, y_val = y[df_all['Cohort']=='Train'], y[df_all['Cohort']=='Val']


#####################################################
# Word Embeddings                                   #
#####################################################
logger.info(
    "Creating embedding matrix using pre-trained w2v vectors."
)

# Now, we need to build the embedding matrix. For this we use
# a pretrained (on the wikipedia corpus) 100-dimensional GloVe
# model.  # to update

# Load the embeddings from a file
embeddings = {}

with open(os.path.join(data_dir, "w2v_reports_128.vec"),
          encoding='utf-8') as file:  # imdb_w2v.txt . w2v_reports_128.vec

#with open('w2v_reports_128.vec',
#          encoding='utf-8') as file:  # imdb_w2v.txt . w2v_reports_128.vec
    dummy = file.readline()
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')

        embeddings[word] = coefs

#embeddings['fatigue']
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
i = 0

#params={'l2':  0.3, 'dropout' : 0.5 , 'lr' : 0.004}

def create_model(params):
    global i
    i += 1
    logger.info('Model' + str(i) + 'training')

    l2 = params['l2']
    dropout = params['dropout']
    lr = params['lr']

    han_model = HAN(
        MAX_WORDS_PER_SENT, MAX_SENT, 1, embedding_matrix,  # 1 is output size
        word_encoding_dim,
        sentence_encoding_dim,
        l1,  
        l2,  
        dropout)

    han_model.summary()

    opt = Adam(lr=lr)

    han_model.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=['acc', rec_scorer, f1_scorer, f2_scorer])

    han_model.fit(X_train, y_train, validation_split=0, batch_size=8,
                  epochs=Nepochs)  # ,callbacks=[es, mc])

    scores = han_model.evaluate(X_val, y_val, verbose=0)

    #f2 = scores[4]
    f1 = scores[3]
    #rec = scores[2]
    #accuracy = scores[1]
    loss = scores[0]
    
    
    # build and save results and parameters
    df_scores=pd.DataFrame([scores],columns=('loss','accuracy','recall','f1','f2'))
      
    df_params=pd.DataFrame.from_dict([params])
  
    df_new=df_scores.join(df_params)
    
    df_results=pd.read_csv(os.path.join(results_dir, out_file+'all.csv'))
    df_results=df_results.append(df_new)
    df_results.to_csv(os.path.join(results_dir, out_file+'all.csv'), index=False)
   
    # Save the best model
    if f1 <= df_results['f1'].max():
        # han_model.save("han_model.hd5")
        print("Save model")
        han_model.save(os.path.join(results_dir, out_file+'_model.hd5'))

    return {'loss': loss, 'params': params, 'status': STATUS_OK}


space = {
    'l2': hp.qloguniform('l2', np.log(0.00001), np.log(0.01), 0.00001),
    'dropout': hp.quniform('dropout', 0, 0.5, 0.2),
    'lr': hp.qloguniform('lr', np.log(0.00001),  np.log(0.05), 0.00001)
}


    
# Trials object to track progress
bayes_trials = Trials()

# File to save first results
try:
    with open(os.path.join(results_dir, out_file+'all.csv'),"r") as f:
        df_results=f.read()
except FileNotFoundError:
    df_results = pd.DataFrame(columns=('loss','accuracy','recall','f1','f2','l2','dropout','lr'))
    df_results.to_csv(os.path.join(results_dir, out_file+'all.csv'), index=False)
 
# Optimize
best = fmin(fn=create_model, space=space, algo=tpe.suggest, max_evals=MAX_EVALS,
            trials=bayes_trials)

print(best)
logger.info('END')

