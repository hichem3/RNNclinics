# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 09:45:49 2019

@authors: Loic Verlingue, Enrico Sartor

Script to use after hyperopt training for hyperparameter search.

"""



################
# directories
################
out_file = 'HAN_30epoch10eval'
data_dir="data/"
results_dir="results/"
scripts_dir="scripts/"

#################
# libraries
#################

import os, re, sys, csv, logging
import seaborn as sns

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import metrics

#from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.layers import (
    Dense, GRU, TimeDistributed, Input,
    Embedding, Bidirectional, Lambda
)
from keras.models import Model
# from keras_han.layers import AttentionLayer
from keras.models import load_model
from keras import regularizers

from sklearn.model_selection import train_test_split, RandomizedSearchCV, \
    GridSearchCV, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import make_scorer, f1_score, recall_score, fbeta_score,\
    precision_recall_fscore_support, accuracy_score, precision_score, \
    confusion_matrix, roc_curve, roc_auc_score, \
    precision_recall_curve, average_precision_score, auc, classification_report
    
from nltk.tokenize import sent_tokenize
#import nltk
#nltk.download('punkt')

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from inspect import signature

import pickle

os.chdir(scripts_dir)
from han_model import HAN
from han_model import AttentionLayer
from utils import rec_scorer, f1_scorer, f2_scorer
os.chdir(results_dir)

#################
# hyperparameters
#################
trained_params=pd.read_csv(os.path.join(results_dir, out_file+"all.csv"), encoding='utf-8') # check
##trained_params=pd.DataFrame({'l1':[0.1,0.3],'l2':[0.2,0.1],'f1':[0.9,0.96]})
trained_params=trained_params[trained_params['f1']==trained_params['f1'].max()]

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
        'Nepochs' : 20,
        'lr':0.001
        }

hp_names=trained_params.columns.values[trained_params.columns.values!='f1']

for key in hp_names:
    params[key]=trained_params[key].iloc[0]

#################
# Data
#################

df_all=pd.read_csv(os.path.join(data_dir, "df_all.csv"), encoding='utf-8')
df_all=df_all[df_all['CompleteValues']]

# Transform the labels into a format Keras can handle
labels = df_all['screenfail']
y = np.asarray(labels)  # to_categorical(labels)

texts = df_all['value']
texts = pd.Series.tolist(texts)
split = df_all['Cohort']

#####################################################
# Tokenization                                      #
#####################################################

# Build a Keras Tokenizer that can encode every token
# saving tokeniser
try:    
    with open(os.path.join(results_dir,out_file+'word_tokeniser'), 'rb') as handle:
        word_tokenizer = pickle.load(handle)
except FileNotFoundError:
    print("Tokeniser not found")
    
'''
word_tokenizer = Tokenizer(num_words=params['max_words'])
word_tokenizer.fit_on_texts(texts)

# saving tokeniser # here ok?
with open(os.path.join(results_dir,'word_tokeniser'), 'wb' ) as handle:
    pickle.dump(word_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
# Construct the input matrix. This should be a nd-array of
# shape (n_samples, MAX_SENT, MAX_WORDS_PER_SENT).
# We zero-pad this matrix (this does not influence
# any predictions due to the attention mechanism.
X = np.zeros((len(texts),params['MAX_SENT'], params['MAX_WORDS_PER_SENT']), dtype='int32')

#review=texts[3]
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


# select the test cohort
X_train,X_test = X[df_all['Cohort']!='Test'], X[df_all['Cohort']=='Test']
y_train, y_test = y[df_all['Cohort']!='Test'], y[df_all['Cohort']=='Test']

#####################################################
# Word Embeddings                                   #
#####################################################

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
# model
#################
'''
han_model = HAN(
        params['MAX_WORDS_PER_SENT'],params['MAX_SENT'], 1, embedding_matrix,  # 1 is output size
        params['word_encoding_dim'],
        params['sentence_encoding_dim'],
        params['l1'],  
        params['l2'],  
        params['dropout'])

han_model.summary()
'''

#### or load model
print("loadmodel")
han_model = load_model(os.path.join(results_dir,out_file+'_model.hd5'), 
                       custom_objects={'HAN': HAN,'AttentionLayer': AttentionLayer, 
                                       'rec_scorer':rec_scorer, 'f1_scorer':f1_scorer, 
                                       'f2_scorer':f2_scorer})

han_model.summary()

print("compile")
################################
# optionaly train / finetune it
han_model.compile(
    optimizer=Adam(lr=params['lr']), loss='binary_crossentropy',
    metrics=['acc']
)


"""
checkpoint_saver = ModelCheckpoint(
    filepath='./tmp/model.{epoch:02d}-{val_loss:.2f}.hdf5',
    verbose=1, save_best_only=True
)
"""
print("train")
history=han_model.fit(X_train, y_train, validation_split=0.1, batch_size=8,
                  epochs=params['Nepochs']) 
 
han_model.save(os.path.join(results_dir, out_file+'_finetuned_model.hd5'))
 
################################
# check results

y_pred_num = han_model.predict(X_test)

### curves
precision, recall, _ =precision_recall_curve(y_test,y_pred_num)
average_precision = average_precision_score(y_test,y_pred_num)
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.figure(1)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
#plt.savefig('AP.pdf')
plt.savefig(os.path.join(results_dir,out_file+'_finetuned_AP.pdf'))

# Compute ROC curve and ROC area for each class
fpr,tpr,_ =roc_curve(y_test,y_pred_num,drop_intermediate=False)
roc_auc = auc(fpr,tpr)
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_num.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw = 2
plt.figure(2)
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
#plt.savefig('ROC.pdf')
plt.savefig(os.path.join(results_dir,out_file+'_finetuned_ROC.pdf'))

# Confusion Matrix - Test data
# bianrize

threshold = 0.5

for i in range(len(y_pred_num)):
    if y_pred_num[i]<threshold:
        y_pred_num[i]=0
    else:
        y_pred_num[i]=1
        

# Using metrics.confusion_matrix function
cm = confusion_matrix(y_test, y_pred_num)
data = cm.tolist()
print("cm returned from sklearn:", data)

#plot cm

plt.figure(0)
ax= plt.subplot()
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');
#ax.add_legend(title="Metrics",  legend_data= [f1,rec,acc]);
#add_legend(title="Metrics",  legend_data= [f1,rec,acc])
sns_plot=sns.heatmap(cm, annot=True, ax = ax,cmap="YlGnBu",cbar =False)
fig = sns_plot.get_figure()
#fig.savefig("cm.pdf")
fig.savefig(os.path.join(results_dir,out_file+'_finetuned_cm.pdf'))

# save metrics
report=classification_report(y_test,y_pred_num,output_dict=True)
report=report['weighted avg']
report['average_precision']=average_precision
report['auc']=roc_auc

#pd.DataFrame([report]).to_csv("report.csv")
pd.DataFrame([report]).to_csv(os.path.join(results_dir,out_file+'_finetuned_metrics.csv'))


'''
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
#fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.legend(loc='best')
plt.show()

plt.savefig(os.path.join(results_dir,out_file+'.pdf'))
'''


'''
threshold = 0.1
y_pred = han_model.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i]<threshold:
        y_pred[i]=0
    else:
        y_pred[i]=1
        
print('y_pred:',y_pred)
print('y_test',y_test)
# Confusion Matrix - Test data
# Using metrics.confusion_matrix function
cm = confusion_matrix(y_test, y_pred)
data = cm.tolist()
print("cm returned from sklearn:", data)
'''
'''
acc = history.history['acc']
val_acc = history.history['val_acc']
rec = history.history['rec']
val_rec = history.history['val_rec']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('HAN_acc_228okk.png')
plt.figure()
plt.plot(epochs, rec, 'bo', label='Training rec')
plt.plot(epochs, val_rec, 'b', label='Validation rec')
plt.title('Training and validation recall')
plt.legend()
plt.savefig('HAN_rec_228okk.png')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('HAN_loss_228okk.png')

f=open('out_HAN_228okk.txt','w')
temp=''
for i in range(len(acc)):
    temp+='At epoch:'+str(i)+'.....Accuracy is:'+str(acc[i])+'Validation accuracy is:'+str(val_acc[i])
    temp+='\n'
f.write(temp)
f.close()


'''
