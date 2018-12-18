from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model, Sequential
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras import layers
from keras.datasets import imdb



max_features = 2000 #10000
# Cut texts after this number of words (among top max_features most common words)
maxlen = 200 #500


# Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
#I have 20000 train samples and 5000 validation because of validation_split set up to 0.2


# Pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


#model definition
model = Sequential()
model.add(layers.Embedding(max_features, 16)) #32
model.add(layers.Dropout(0.5))
model.add(layers.Bidirectional(layers.LSTM(16))) #32
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2) #epochs=10,batch=128



history_dict = history.history 
history_dict.keys() #['acc', 'loss', 'val_acc', 'val_loss']

#plotting training and validation LOSS
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = [i for i in range(1, len(history_dict['acc']) + 1)]
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plotting training and validation ACCURACY
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#now I should run a prediction on the x_test
