# Author: Enrico Sartor, Loic Verlingue

from WorkflowSF import HAN
#from WorkflowSF import han_model
from WorkflowSF import AttentionLayer
import keras
from keras.models import load_model
from WorkflowSF.utils import rec_scorer                                     

from keras import backend as K
from keras.layers import (
    Dense, GRU, TimeDistributed, Input,
    Embedding, Bidirectional, Lambda, Dropout
)
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers

#
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


#####################################################
# building model and loading weights or load custom objects from saved models
#####################################################

# build and load wieghts

logger.info("Building the model.")

han_model = HAN(
    MAX_WORDS_PER_SENT, MAX_SENT, 1, embedding_matrix,
    word_encoding_dim, sentence_encoding_dim,
    l1,l2,dropout
)

han_model.summary()

#load weights
# todo

'''
han_model.compile(
    optimizer=Adam(lr=0.0001), loss='binary_crossentropy',
    metrics=['acc',rec]
)
"""
checkpoint_saver = ModelCheckpoint(
    filepath='./tmp/model.{epoch:02d}-{val_loss:.2f}.hdf5',
    verbose=1, save_best_only=True
)
"""
history = han_model.fit(
    X_train, y_train, batch_size=batch_size, epochs=50,
    validation_split = 0.2,
    #callbacks=[checkpoint_saver]
)
'''
#####
# of loading model directly
logger.info("Load the model.")

han_model = load_model({file_path}, custom_objects={
    'HAN': HAN,
    'AttentionLayer': AttentionLayer
})

#han_model=load_model(os.path.join(results_dir, out_file+'_model.hd5'), custom_objects={'AttentionLayer': AttentionLayer, 'HAN':HAN})
#todo

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
