# Text pre processing performed to train HAN
# Author: Enrico Sartor
# path on venv with X_REP_RAW cocrresponding to a file with subdirectories: pos and neg corresponding to med reports with positive or negative labels

#####################################################
# Pre processing                                    #
#####################################################
logger.info("Pre-processing data.")

imdb_dir = 'X_REP_RAW'
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(imdb_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname),encoding='utf-8',errors='ignore')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

#####################################################
# Tokenization                                      #
#####################################################
logger.info("Tokenization.")

# Build a Keras Tokenizer that can encode every token
word_tokenizer = Tokenizer(num_words=MAX_VOC_SIZE)
word_tokenizer.fit_on_texts(texts)

# Construct the input matrix. This should be a nd-array of
# shape (n_samples, MAX_SENT, MAX_WORDS_PER_SENT).
# We zero-pad this matrix (this does not influence
# any predictions due to the attention mechanism.
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

# Transform the labels into a format Keras can handle
y = np.asarray(labels)
# We make a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)

#####################################################
# Word Embeddings                                   #
#####################################################
logger.info(
    "Creating embedding matrix using pre-trained w2v vectors."
)

# Now, we need to build the embedding matrix. For this we use
# a pretrained (on the wikipedia corpus) 100-dimensional GloVe
# model.

# Load the embeddings from a file
embeddings = {}
#r'C:\Users\Enrico\Desktop\Projet Innovation\
with open('w2v_reports_128.vec', encoding='utf-8') as file:
    dummy = file.readline()
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')

        embeddings[word] = coefs

# Initialize a matrix to hold the word embeddings
embedding_matrix = np.random.random(
    (len(word_tokenizer.word_index) + 1, GLOVE_DIM)
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

