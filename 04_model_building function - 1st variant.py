
# coding: utf-8

# <h3> Model Building function - variante 1 <h3>

# In[ ]:



def model_building(Tx, Ty, n_a, n_s, human_vocab_size,machine_vocab_size):
    
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx,human_vocab_size))
    
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    # Initialize empty list of outputs
    outputs_dense = [] #This will become a 10-characters string of 0s and 1s, 'translated' iteratively,
    #we are interested only in the last predicted character 
    s_dense = s0

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True.
    #return_sequences: Whether to return the last output in the output sequence, or the full sequence.
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X) #ENCODER
    
    # Step 2: Iterate for Ty steps
    # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t
    for t in range(10): #I choose to have a many to many Tx-->10
        context = one_step_attention(a, s) #output is [?,1,64]
        
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state]
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])  #DECODER
        
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM
        #out = output_layer(s) (output_layer =  Dense(len(machine_vocab), activation=softmax))
        s_dense = output_layer(s)
        
        # Step 2.D: Append "out" to the "outputs" list
        #outputs.append(out)
        outputs_dense.append(s_dense) 
        
   
    # Step 3: Create model instance taking three inputs and returning the list of outputs.
    model = Model(inputs=[X, s0, c0], outputs=outputs_dense[-1])

    return model

