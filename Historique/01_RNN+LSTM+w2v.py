import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from keras.datasets import imdb

import logging
from gensim.models import Word2Vec, KeyedVectors
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 

(x_train, y_train), (x_test, y_test) = imdb.load_data()
sentences = x_train


model_2 = Word2Vec(size=100, min_count=1)
#size= Dimensionality of the word vectors.
#min_count=Ignores all words with total frequency lower than this.
model_2.build_vocab(sentences)
total_examples = model_2.corpus_count
model = KeyedVectors.load_word2vec_format(r"C:\Users\Enrico\Desktop\Projet Innovation\GoogleNews-vectors-negative300.bin", binary=True)
model_2.build_vocab([list(model.vocab.keys())], update=True)
model_2.intersect_word2vec_format(r"C:\Users\Enrico\Desktop\Projet Innovation\GoogleNews-vectors-negative300.bin", binary=True)
model_2.train(sentences, total_examples=total_examples, epochs=model_2.iter)
