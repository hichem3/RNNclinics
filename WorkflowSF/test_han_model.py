'''
loads and perform prediction with pre-trained HAN (hierachical attention network, specialized neural net for NLP tasks) models
Authors: Loic Verlingue, Enrico Sartor, Valentin Charvet, Hichem Aloui

Requirements:
needs in data_dir: 
    w2v_reports_128.vec (word2vec. Dictionnaire vectors representing each words)
    word_tokeniser, out_file+'_model.hd5' # optionnaly df_all.csv (hyperopt training results), if "load test cohort" active
needs in scripts_dir: utils.py, han_model.py used to build the HAN model
'''

################
# directories
################
out_file = 'HAN_30epoch10eval'
data_dir="data/" #pool results of hyperopt and finetune in data
scripts_dir="scripts/"

#################
# libraries
#################

import os
import pandas as pd
import numpy as np
import keras
#from keras import backend as K
from keras.layers import (
    Dense, GRU, TimeDistributed, Input,
    Embedding, Bidirectional, Lambda, Dropout
)
#from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras import regularizers
from keras.models import load_model
#from keras.optimizers import Adam

from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

#from sklearn.metrics import confusion_matrix

import pickle # to save and load keras tokeniser

os.chdir(scripts_dir)
from han_model import HAN
from han_model import AttentionLayer
from utils import rec_scorer, f1_scorer, f2_scorer
os.chdir(data_dir)

#################
# hyperparameters
#################
#trained_params=pd.read_csv(os.path.join(data_dir, out_file+"all.csv"), encoding='utf-8') # check
##trained_params=pd.DataFrame({'l1':[0.1,0.3],'l2':[0.2,0.1],'f1':[0.9,0.96]})
#trained_params=trained_params.loc[trained_params['f1']==trained_params['f1'].max(),]

params={
        'MAX_WORDS_PER_SENT' : 40,
        'MAX_SENT' : 80,
        'max_words' : 10000,
        'embedding_dim' : 128,
        'word_encoding_dim':256,
        'sentence_encoding_dim':256,
        'l1':0,
        'l2':0,
        'dropout':0.2,
        'MAX_EVALS' : 10, # number of models to evaluate with hyperopt
        'Nepochs' : 100,
        'lr':0.001
        }

# replace by best trained parameters
#hp_names=trained_params.columns.values[trained_params.columns.values!='f1']

#for key in hp_names:
#    params[key]=trained_params[key]


#####################################################
# Word Embeddings                                   #
#####################################################

# Now, we need to build the embedding matrix. For this we use
# a pretrained (on the wikipedia corpus) 100-dimensional GloVe
# model.  # to update

# loading tokeniser
with open(os.path.join(data_dir,'word_tokeniser'), 'rb') as handle:
    word_tokenizer = pickle.load(handle)

# Load the embeddings from a file
embeddings = {}

with open(os.path.join(data_dir, "w2v_reports_128.vec"),
          encoding='utf-8') as file:  # imdb_w2v.txt . w2v_reports_128.vec
    dummy = file.readline()
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')

        embeddings[word] = coefs

#embeddings['fatigue']
# Initialize a matrix to hold the word embeddings
embedding_matrix = np.random.random(
    (len(word_tokenizer.word_index) + 1, params['embedding_dim'])
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

#################
# New Data
#################

# entry
text1 = ['Va bien. Examen clinique normal. Signature du consentement']
text2 = ['Ne va pas bien. Examen clinique anormal. Signes d insuffisance cardique. Attente d echographie.']

texts=text2
# to debug format
#texts=pd.DataFrame({'review':[str(text1),str(text2)]})

# or load test cohort
'''
df_all=pd.read_csv(os.path.join(data_dir, "df_all.csv"), encoding='utf-8')
df_all=df_all[df_all['CompleteValues']]

labels = df_all['screenfail']

texts = df_all['value']
texts = texts[df_all['Cohort']=='Test']
texts = pd.Series.tolist(texts)

# select the test cohort
# Transform the labels into a format Keras can handle
y = np.asarray(labels)  # to_categorical(labels)
y_test = y[df_all['Cohort']=='Test']
'''

#####################################################
# Tokenization                                      #
#####################################################

# Construct the input matrix. This should be a nd-array of
# shape (n_samples, MAX_SENT, MAX_WORDS_PER_SENT).
# We zero-pad this matrix (this does not influence
# any predictions due to the attention mechanism.
X = np.zeros((len(texts),params['MAX_SENT'], params['MAX_WORDS_PER_SENT']), dtype='int32')

i=0
review=texts[0]

for i, review in enumerate(texts):
    sentences = sent_tokenize(review)
    tokenized_sentences = word_tokenizer.texts_to_sequences(
        sentences
    )
    tokenized_sentences = pad_sequences(
        tokenized_sentences, maxlen=params['MAX_WORDS_PER_SENT']
    )

    pad_size = params['MAX_SENT'] - tokenized_sentences.shape[0]

    if pad_size < 0:
        tokenized_sentences = tokenized_sentences[0:params['MAX_SENT']]
    else:
        tokenized_sentences = np.pad(
            tokenized_sentences, ((0, pad_size), (0, 0)),
            mode='constant', constant_values=0
        )

    # Store this observation as the i-th observation in
    # the data matrix
    X[i] = tokenized_sentences[None, ...]

X
#################
# model
#################

han_model = load_model(os.path.join(data_dir,out_file+'_model.hd5'), 
                       custom_objects={'HAN': HAN,'AttentionLayer': AttentionLayer, 
                                       'rec_scorer':rec_scorer, 'f1_scorer':f1_scorer, 
                                       'f2_scorer':f2_scorer})

han_model.summary()

################################
# predict

prediction = han_model.predict(X)
# output
print(prediction)
