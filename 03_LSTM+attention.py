from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import numpy as np
from keras.activations import softmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.datasets import imdb
#%matplotlib inline

###############################################################################
#Parameters set-up
###############################################################################
maxlen = 20 #cutoff reviews, raise to 500
training_samples = 50 #raise to 20000
val = 10 #raise to 5000
test=val = 10
max_words = 200 #vocabulary dimension, raise to 10000
machine_vocab_size=2
Tx=maxlen
Ty=1
m=training_samples #len(x_train)

###############################################################################
#Methods set-up
###############################################################################
# Defined shared layers as global variables
repeator = RepeatVector(Tx) 
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh") # check N units
densor2 = Dense(1, activation = "relu") # check N units
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)
n_a = 32 #dimension of LSTM hidden states for the BiLSTM
n_s = 64 #dimension of LSTM hidden states for the post-LSTM
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(machine_vocab_size, activation='sigmoid')
s0=np.zeros((m,n_s)) 
c0=np.zeros((m,n_s))

###############################################################################
#Functions set-up
###############################################################################
def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a"
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis
    concat = concatenator([a, s_prev])# [?,5000,64] U [?,5000,64]
    #print('concat.shape',concat.shape) #(?, 5000, 128)
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e
    e = densor1(concat)
    #print('e.shape',e.shape)#(?, 5000, 10)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies.
    energies = densor2(e)
    #print('energies',energies.shape)# (?, 5000, 1)
    # Use "activator" on "energies" to compute the attention weights "alphas"
    alphas = activator(energies)
    #print('alphas',alphas.shape)#(?, 5000, 1)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell
    context = dotor([alphas, a])  #dot product alphas.transpose*a : 1x5000 * 5000x64 
    #print('context',context.shape) #context (?, 1, 64)
    return context


def model_building(Tx, Ty, n_a, n_s, human_vocab_size,machine_vocab_size):
    
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence or LSTM units
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab" # to be
    changed into patient's documents
    machine_vocab_size -- size of the python dictionary "machine_vocab" # to
    be changed into sigmoid function

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx,human_vocab_size))
    #print(X)
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    # Initialize empty list of outputs
    outputs = []

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True.
    #return_sequences: Whether to return the last output in the output sequence, or the full sequence.
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X) #ENCODER
    # Step 2: Iterate for Ty steps
    # no need if only 1 logistic output
    for t in range(Ty):
    # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t

        context = one_step_attention(a, s) #output is [?,1,64]
        
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state]
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])  #DECODER
        #print('s',s) #[?,64]
        #print('_',_) #[?,64]
        #print('c',c) #[?,64]
        
        # end of loop here ? to finish with single output
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM
        #out = output_layer(s) (output_layer =  Dense(len(machine_vocab), activation=softmax))
        s = output_layer(s)
        # Step 2.D: Append "out" to the "outputs" list
        #outputs.append(out)
        outputs.append(s)
    
    #print(len(outputs[:])) #Ty
    
    # Step 3: Create model instance taking three inputs and returning the list of outputs.
    model = Model(inputs=[X, s0, c0], outputs=outputs)

    return model

###############################################################################
#Get the datas
###############################################################################
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_words)
word_index = imdb.get_word_index()

###############################################################################
#Vectorizations
###############################################################################
def vectorize_sequences(sequences, dimension=max_words):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


x_train = vectorize_sequences(train_data[:training_samples])
x_val= vectorize_sequences(train_data[training_samples:training_samples+val])
x_test = vectorize_sequences(test_data[:test])

y_train = np.asarray(train_labels[:training_samples]).astype('float32').reshape(training_samples,1)
y_val= np.asarray(train_labels[training_samples:training_samples+val]).astype('float32').reshape(val,1)
y_test = np.asarray(test_labels[:test]).astype('float32').reshape(test,1)
X = np.array(list(map(lambda x: to_categorical(x, num_classes=maxlen), x_train))) 
Y = np.array(list(map(lambda x: to_categorical(x, num_classes=machine_vocab_size), y_train)))

X=X.swapaxes(1,2) #50x20x200
out=list(Y.swapaxes(0,1)) #1x50x2

###############################################################################
#Model definition
###############################################################################
model = model_building(Tx, Ty, n_a, n_s, max_words,machine_vocab_size)

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acc'])

history = model.fit([X,s0,c0],out,epochs=4,batch_size=16)#,validation_data=(x_val, y_val)) #increase batch size, epochs should be 10-17

###############################################################################
#Validation
###############################################################################
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
