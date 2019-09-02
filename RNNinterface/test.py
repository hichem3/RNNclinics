
def pred(mytext):

    import os
    import pickle

    from han_model import HAN
    from han_model import AttentionLayer
    from utils import rec_scorer, f1_scorer, f2_scorer




    import numpy as np

    from keras.preprocessing.sequence import pad_sequences


    from keras.models import load_model



    from nltk.tokenize import sent_tokenize
    import nltk
    nltk.download('punkt')



    out_file = 'HAN_100epoch10eval'
    data_dir = "C:/Users/hichem/PycharmProjects/phase1_prediction"
    scripts_dir = "C:/Users/hichem/PycharmProjects/phase1_prediction"

    os.chdir(scripts_dir)
    os.chdir(data_dir)
    params = {
        'MAX_WORDS_PER_SENT': 40,
        'MAX_SENT': 80,
        'max_words': 10000,
        'embedding_dim': 128,
        'word_encoding_dim': 256,
        'sentence_encoding_dim': 256,
        'l1': 0,
        'l2': 0,
        'dropout': 0.2,
        'MAX_EVALS': 10,  # number of models to evaluate with hyperopt
        'Nepochs': 100,
        'lr': 0.001
    }



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
    # replace the random initalisation with the GloVe vector.
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


    texts = mytext




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




    #################
    # model
    #################

    han_model = load_model(os.path.join(data_dir,out_file+'_model.hd5'),
                           custom_objects={'HAN': HAN,'AttentionLayer': AttentionLayer,
                                           'rec_scorer':rec_scorer, 'f1_scorer':f1_scorer,
                                          'f2_scorer':f2_scorer})



    #han_model.summary()

    ################################
    # predict

    prediction = han_model.predict(X)

    # output

    print(prediction)
    return (prediction)

