# Text pre processing performed to train HAN
# Author: Enrico Sartor, Loic Verlingue

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
# Pre processing                                    #
#####################################################

logger.info("Pre-processing data.")

# Loic version

df_all=pd.read_csv(os.path.join(data_dir, "df_all.csv"), encoding='utf-8')
df_all=df_all[df_all['CompleteValues']]

labels = df_all['screenfail']

texts = df_all['value']
texts = pd.Series.tolist(texts)
split = df_all['Cohort']


"""
logger.info("Pre-processing Kaggle's data.")

# Load Kaggle's IMDB example data
data = pd.read_csv('labeledTrainData.tsv', sep='\t')


# Do some basic cleaning of the review text
def remove_quotations(text):

    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    return text


def remove_html(text):

    tags_regex = re.compile(r'<.*?>')
    return tags_regex.sub('', text)


data['review'] = data['review'].apply(remove_quotations)
data['review'] = data['review'].apply(remove_html)
data['review'] = data['review'].apply(lambda x: x.strip().lower())

# Get the data and the sentiment
texts = data['review'].values
labels = data['sentiment'].values
del data
"""

#####################################################
# Tokenization                                      #
#####################################################
logger.info("Tokenization.")

# Build a Keras Tokenizer that can encode every token
word_tokenizer = Tokenizer(num_words=max_words)
word_tokenizer.fit_on_texts(texts)

# Construct the input matrix. This should be a nd-array of
# shape (n_samples, MAX_SENT, MAX_WORDS_PER_SENT).
# We zero-pad this matrix (this does not influence
# any predictions due to the attention mechanism.
X = np.zeros((len(texts), MAX_SENT, MAX_WORDS_PER_SENT), dtype='int32')

#review=texts[3]
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
            tokenized_sentences, ((0, pad_size), (0, 0)),
            mode='constant', constant_values=0
        )

    # Store this observation as the i-th observation in
    # the data matrix
    X[i] = tokenized_sentences[None, ...]

# Transform the labels into a format Keras can handle
y = np.asarray(labels)  # to_categorical(labels)

# We make a train/test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)

X_train, X_test = X[df_all['Cohort']=='Train'], X[df_all['Cohort']=='Test']
y_train, y_test = y[df_all['Cohort']=='Train'], y[df_all['Cohort']=='Test']

#X_train.shape, y_train.shape

"""
X_train=X_train[:60]
X_test=X_test[:10]
y_train=y_train[:60]
y_test=y_test[:10]
"""

#####################################################
# Word Embeddings                                   #
#####################################################
logger.info(
    "Creating embedding matrix using pre-trained w2v vectors."
)

# Now, we need to build the embedding matrix. For this we use
# a pretrained (on the wikipedia corpus) 100-dimensional GloVe
# model.  # to update

# Load the embeddings from a file
embeddings = {}
# r'C:\Users\Enrico\Desktop\Projet Innovation\

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
    (len(word_tokenizer.word_index) + 1, embedding_dim)
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
