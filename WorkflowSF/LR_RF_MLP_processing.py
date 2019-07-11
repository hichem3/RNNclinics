####################
#  Author: Valentin Charvet
# machine learning and MLP 
# pre processing of the data

### Analysis of text reports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from clintk.utils.connection import get_engine, sql2df
from clintk.utils.unfold import Unfolder
from prescreen.vcare.biology import fetch_and_fold as fetch_bio_vcare
from prescreen.simbad.biology_2 import fetch_and_fold as fetch_bio_simbad

from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE

from sklearn.model_selection import cro
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from clintk.utils.connection import get_engine, sql2df
from clintk.utils.unfold import Unfolder
from prescreen.vcare.biology import fetch_and_fold as fetch_bio_vcare
from prescreen.simbad.biology_2 import fetch_and_fold as fetch_bio_simbad

from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, average_precision_score, \
    classification_report, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from clintk.text2vec.w2v_clusters import WordClustering
from prescreen.vcare.reports import fetch_and_fold as rep_vc
from prescreen.simbad.parse_reports import fetch_and_fold as rep_sb

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch, DBSCAN
from sklearn.decomposition import PCA, NMF

sns.set()

## Fetching data

engine = get_engine('vcare', 'srt-ap-92', 'vcare')

target_vcare = sql2df(engine, 'patient_target')
target_simbad = sql2df(engine, 'patient_target_simbad')

#features = pd.read_csv(feature_path, sep=';')

target_vcare = target_vcare.loc[:, ['nip', 'id', 'screenfail']]
target_simbad.drop(['prescreen', 'index'], axis=1, inplace=True)

target = pd.concat([target_vcare, target_simbad], axis=0, ignore_index=True)
print(target.shape)
target.head()

table, path = 'event', '../data/cr_sfditep_2012.xlsx'

df_rep_vc = rep_vc(table, engine, 'patient_target', 1) # SF only ou tout?
df_rep_sb = rep_sb(path, engine, 'patient_target_simbad', 1)

df_rep_sb.rename({'id': 'patient_id'}, axis=1, inplace=True)

df_rep = pd.concat([df_rep_vc, df_rep_sb], ignore_index=True, sort=False)

df_rep = df_rep.merge(target[['id', 'screenfail']], left_on='patient_id', right_on='id')
df_rep.head()

#
x_rep_raw, y_rep = df_rep['value'], df_rep['screenfail']

#
x_rep_raw.shape,  y_rep.sum() / y_rep.shape[0]
len(x_rep_raw[1])

# CV is then performed with StratifiedShuffleSplit funtion
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.15)
scoring = {'AUC': 'roc_auc', 'Recall': 'recall'}

## then: Searching for best transformer


## loading data for HAN model
# assuming data is ordered, y == y_rep
X, y = np.loadtxt('X_HAN.txt'), np.loadtxt('y_han.txt')
embedding_matrix = np.loadtxt('embedding_matrix_HAN.txt')

# number of features for HAN
p_han = X.shape[1]

# concatenation of reports matrix and HAN feature matrix
X_whole = np.hstack(x_rep_raw, X)

X_train, X_test, y_train, y_test = train_test_split(X_whole, y, test_size=TEST_SPLIT)

train_base, test_base = X_train[:, :-p_han], X_test[:, :-p_han]
train_HAN, test_HAN = X_train[:, -p_han:], X_test[:, -p_han:]

# train HAN model (as in HAN_train_test.py
logger.info("Training the model.")

han_model = HAN(
    MAX_WORDS_PER_SENT, MAX_SENT, 1, embedding_matrix,
    word_encoding_dim, sentence_encoding_dim,
    l1,l2,dropout
)

han_model.summary()

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


# train base model
# TFIDF + RF ?


# evaluation of model (compare which one is best)