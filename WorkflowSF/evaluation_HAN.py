# script to train and evaluate HAN model from data loaded by whole_preprocessing

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import  pad_sequences
from keras.optimizers import Adam
from nltk.tokenize import sent_tokenize
from sklearn.metrics import confusion_matrix, recall_score

from whole_preprocessing import X_train, X_test, y_train, y_test, logger
from HAN_model_building import HAN
from HAN_utils import rec

sns.set()
# preprocessing data
logger.info("Tokenization for HAN model")

#@TODO fill variables
MAX_VOC_SIZE = ..
MAX_SENT = ..
MAX_WORDS_PER_SENT = ..
GLOVE_DIM = ..
n_train = X_train.shape[0]

# concatenating raw texts for tokenization
texts = np.vstack(X_train, X_test)

word_tokenizer = Tokenizer(num_words=MAX_VOC_SIZE)
word_tokenizer.fit_on_texts(texts)

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

X_train_han, X_test_han = X[:n_train, :], X[n_train:, :]

# loading embedding matri
logger.info("Loading embedding matrix")

embeddings = {}
with open('w2v_reports_128.vec', encoding='utf-8') as file:
    dummy = file.readline()
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')

        embeddings[word] = coefs

# Initialize a matrix to hold the word embeddings
embedding_matrix = np.random.random(
    (len(word_tokenizer.word_index) + 1, GLOVE_DIM))

embedding_matrix[0] = 0

# Loop though all the words in the word_index and where possible
# replace the random initalization with the GloVe vector.
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# Training HAN model
logger.info("Training HAN model")

# hyperparameters for the model
word_encoding_dim = 70
sentence_encoding_dim = 50
l1 = 0.0015
l2 = 0.025
dropout = 0.
batch_size = 96

han_model = HAN(
    MAX_WORDS_PER_SENT, MAX_SENT, 1, embedding_matrix,
    word_encoding_dim, sentence_encoding_dim,
    l1,l2,dropout)

han_model.summary()

han_model.compile(
    optimizer=Adam(lr=0.0001), loss='binary_crossentropy',
    metrics=['acc',rec])

"""
checkpoint_saver = ModelCheckpoint(
    filepath='./tmp/model.{epoch:02d}-{val_loss:.2f}.hdf5',
    verbose=1, save_best_only=True
)
"""

history = han_model.fit(
    X_train_han, y_train, batch_size=batch_size, epochs=50,
    validation_split = 0.2)
    #callbacks=[checkpoint_saver]

# check results
logger.info("Evaluation of model ")

threshold = 0.5
y_pred = han_model.predict(X_test_han)
for i in range(len(y_pred)):
    if y_pred[i]<threshold:
        y_pred[i]=0
    else:
        y_pred[i]=1

print('y_pred:',y_pred)
print('y_test',y_test)

print("Recall score with HAN={:.3f}".format(recall_score(y_test, y_pred)))

cm = confusion_matrix(y_test, y_pred)
data = cm.tolist()
print("cm returned from sklearn:", data)

logger.info("Displaying detailed training trajectories")

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