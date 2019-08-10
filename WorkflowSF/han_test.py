# Author: Enrico Sartor, Loic Verlingue

from WorkflowSF import HAN
from WorkflowSF import AttentionLayer

import os
import pandas as pd
import numpy as np
import keras
from keras import backend as K
from keras.layers import (
    Dense, GRU, TimeDistributed, Input,
    Embedding, Bidirectional, Lambda, Dropout
)
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.models import load_model



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
    return (beta_squared + 1) * (precision * recall) / (
                beta_squared * precision + recall + K.epsilon())


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
    return (beta_squared + 1) * (precision * recall) / (
                beta_squared * precision + recall + K.epsilon())


############################################################################
# model path
file_path=.

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


#####
# loading model
logger.info("Load the model.")

han_model = load_model("results/HAN_100epoch10eval_model.hd5", custom_objects={'HAN': HAN,'AttentionLayer': AttentionLayer, 'rec_scorer':rec_scorer, 'f1_scorer':f1_scorer, 'f2_scorer':f2_scorer})

###############################
# load new data


################################
# check results

threshold = 0.5
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
