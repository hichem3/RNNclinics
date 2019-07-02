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
