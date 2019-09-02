####################
#  Author: Valentin Charvet, Loic Verlingue
# machine learning and MLP 
# pre processing of the data

### Analysis of text reports

out_file = 'MLmodels3008'
data_dir="data/"
results_dir="results/"
scripts_dir="scripts/"

import os

import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

#from clintk.utils.connection import get_engine, sql2df
#from clintk.utils.unfold import Unfolder
#from prescreen.vcare.biology import fetch_and_fold as fetch_bio_vcare
#from prescreen.simbad.biology_2 import fetch_and_fold as fetch_bio_simbad

from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, average_precision_score, \
    classification_report, recall_score, f1_score, roc_auc_score, make_scorer, accuracy_score
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator

from gensim.models import Word2Vec, KeyedVectors 
#from clintk.text2vec.w2v_clusters import WordClustering
#from prescreen.vcare.reports import fetch_and_fold as rep_vc
#from prescreen.simbad.parse_reports import fetch_and_fold as rep_sb

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch, DBSCAN
from sklearn.decomposition import PCA, NMF

os.chdir(scripts_dir)

from w2v_clusters import WordClustering

#sns.set()

#####
# data, precomputed with processing, lolad here

df_all=pd.read_csv(os.path.join(data_dir, "df_all.csv"), encoding='utf-8')
df_all=df_all[df_all['CompleteValues']]

#df_all.shape
#df_all.columns.values

labels = df_all['screenfail']
#pd.value_counts(labels)

texts = df_all['value']
texts = pd.Series.tolist(texts)
split = df_all['Cohort']

#x_rep_raw=df_all.loc[split!="Test",'value']
x_rep_raw=df_all[split!="Test"].value.apply(str)
#x_rep_raw.__class__
y_rep=df_all.loc[split!="Test"].screenfail
pd.value_counts(y_rep)

# CV is then performed with StratifiedShuffleSplit funtion
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.15)
scoring = {'accuracy':'accuracy','AUC': 'roc_auc','f1':'f1', 'Recall': 'recall'}

### rf for 2 transfomers
#tfidf = TfidfVectorizer(min_df=0.025, max_df=0.9, ngram_range=(3, 3),
# norm='l2')
#x_tfidf = tfidf.fit_transform(x_rep_raw)


w2v_cluster = WordClustering( pretrained=True,
                              model_path= os.path.join(data_dir,
                                                  "w2v_reports_128.vec"),
                              n_clusters=30,
                             clustering=AgglomerativeClustering())


x_w2v = w2v_cluster.fit(x_rep_raw.apply(lambda s:  s.split(' '))).transform(x_rep_raw.apply(lambda s:  s.split(' ')))

print(w2v_cluster.word_vectors_.shape)


# rf
rf = RandomForestClassifier()

params = {'n_estimators': np.linspace(5, 200, num=50, dtype=int),
         # 'max_features': ['log2', 'sqrt'],
          'min_samples_leaf': np.linspace(0.001, 0.5, num=50)}

grid = GridSearchCV(rf,
                    param_grid=params,
                    cv=cv,
                    scoring=scoring,
                    refit='Recall',
                    return_train_score=False,
                    verbose=1,
                    n_jobs=-1)

#grid.fit(x_tfidf, y_rep)

grid.fit(x_w2v, y_rep)

results = pd.DataFrame(grid.cv_results_)
cols = ['split{}_test_accuracy'.format(i) for i in range(5)] + ['split{}_test_AUC'.format(i) for i in range(5)] + ['split{}_test_Recall'.format(i) for i in range(5)] + ['split{}_test_f1'.format(i) for i in range(5)]
results.drop(columns=cols, inplace=True)
results.sort_values(['rank_test_f1'], inplace=True)
results.to_csv(os.path.join(results_dir, out_file+"rf_gridsearch.csv"))
results.shape

### pipeline with RF vs LR in a single CV
clf = LogisticRegression(penalty='l1')

'''
params = dict(clf=[RandomForestClassifier(50, max_features='auto'),
                   LogisticRegression(penalty='l1')],
                  clf__C=[np.logspace(0.1, 0.9, num=5)],
                  clf__n_estimators=[np.linspace(10, 100, num=5, dtype=int)])
'''            
#params = dict(C=[0.1, 0.2, 0.3, 0.4, 0.5])
params = {'C':np.linspace(0.001, 0.9, num=100)}

grid_search = GridSearchCV(clf, 
                           param_grid=params,
                           cv=cv,
                    scoring=scoring,
                    refit='f1',
                    return_train_score=False,
                    verbose=1,
                    n_jobs=-1)

#grid_search.fit(x_tfidf, y_rep)
grid_search.fit(x_w2v, y_rep)

results = pd.DataFrame(grid_search.cv_results_)
cols = ['split{}_test_accuracy'.format(i) for i in range(5)] + ['split{}_test_AUC'.format(i) for i in range(5)] + ['split{}_test_Recall'.format(i) for i in range(5)] + ['split{}_test_f1'.format(i) for i in range(5)]
results.drop(columns=cols, inplace=True)
results.sort_values(['rank_test_f1'], inplace=True)
results.to_csv(os.path.join(results_dir, out_file+"LR_gridsearch.csv"))
results.shape
