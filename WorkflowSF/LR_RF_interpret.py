# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:28:59 2019

script to intepret models decisions

UNTESTED

@author: Valentin Charvet, Loic Verlingue
"""

out_file = 'MLmodels.csv'
data_dir="data/"
results_dir="Results/"
scripts_dir="scripts/"


# check imports
import os

import pandas as pd
import numpy as np
import seaborn as sns
#import matplotlib.pyplot as plt

#from clintk.utils.connection import get_engine, sql2df
#from clintk.utils.unfold import Unfolder
#from prescreen.vcare.biology import fetch_and_fold as fetch_bio_vcare
#from prescreen.simbad.biology_2 import fetch_and_fold as fetch_bio_simbad

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
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, f1_score, recall_score, fbeta_score,\
    precision_recall_fscore_support, accuracy_score, precision_score, \
    confusion_matrix, roc_curve, roc_auc_score, \
    precision_recall_curve, average_precision_score, auc, classification_report

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch, DBSCAN
from sklearn.decomposition import PCA, NMF
    
from gensim.models import Word2Vec, KeyedVectors 
#from clintk.text2vec.w2v_clusters import WordClustering
#from prescreen.vcare.reports import fetch_and_fold as rep_vc
#from prescreen.simbad.parse_reports import fetch_and_fold as rep_sb

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from inspect import signature

os.chdir(scripts_dir)
from w2v_clusters import WordClustering

sns.set()

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
y_rep=df_all.loc[split!="Test"].screenfail

x_test=df_all[split=="Test"].value.apply(str)
y_test=df_all.loc[split=="Test"].screenfail

##############
# embedding

w2v_cluster = WordClustering( pretrained=True,
                              model_path= os.path.join(data_dir,
                                                  "w2v_reports_128.vec"),
                              n_clusters=30,
                             clustering=AgglomerativeClustering())


x_w2v = w2v_cluster.fit(x_rep_raw.apply(lambda s:  s.split(' '))).transform(x_rep_raw.apply(lambda s:  s.split(' ')))

w2v_cluster.word_vectors_.shape

#
words_embedded = TSNE().fit_transform(w2v_cluster.word_vectors_)

plt.figure(figsize=(12, 12))
plt.scatter(words_embedded[:, 0], words_embedded[:, 1],
            c=w2v_cluster.cluster_ids_, s=10, alpha=0.8)
plt.savefig(os.path.join(results_dir,out_file+'_w2v_clusters.pdf'))


#feature_names = tfidf.get_feature_names() # waht about for v2w? same?
print(feature_names)

###############
# RF

## ! todo: set best hyperparameters from search
rf = RandomForestClassifier(n_estimators=16, min_samples_leaf=0.001, max_features='sqrt', max_depth=6)
rf.fit(x_w2v, y_rep)

# LR
LR = LogisticRegression(C=0.890919191919192, penalty='l1')
LR.fit(x_w2v, y_rep)

###############
# evaluation        ##!tocheck ##!todo for LR
x_w2v_test = w2v_cluster.fit(x_test.apply(lambda s:  s.split(' '))).transform(x_test.apply(lambda s:  s.split(' ')))

y_pred_num = LR.predict(x_w2v_test)

### curves
precision, recall, _ =precision_recall_curve(y_test,y_pred_num)
average_precision = average_precision_score(y_test,y_pred_num)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.figure(1)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
#plt.savefig('AP.pdf')
plt.savefig(os.path.join(results_dir,out_file+'_rf_AP.pdf'))

# Compute ROC curve and ROC area for each class
fpr,tpr,_ =roc_curve(y_test,y_pred_num,drop_intermediate=False)
roc_auc = auc(fpr,tpr)
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_num.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw = 2
plt.figure(2)
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
#plt.savefig('ROC.pdf')
plt.savefig(os.path.join(results_dir,out_file+'_rf_ROC.pdf'))

# Confusion Matrix - Test data
# bianrize

threshold = 0.5

for i in range(len(y_pred_num)):
    if y_pred_num[i]<threshold:
        y_pred_num[i]=0
    else:
        y_pred_num[i]=1
        

# Using metrics.confusion_matrix function
cm = confusion_matrix(y_test, y_pred_num)
data = cm.tolist()
print("cm returned from sklearn:", data)

#plot cm

plt.figure(0)
ax= plt.subplot()
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');
#ax.add_legend(title="Metrics",  legend_data= [f1,rec,acc]);
#add_legend(title="Metrics",  legend_data= [f1,rec,acc])
sns_plot=sns.heatmap(cm, annot=True, ax = ax,cmap="YlGnBu",cbar =False)
fig = sns_plot.get_figure()
#fig.savefig("cm.pdf")
fig.savefig(os.path.join(results_dir,out_file+'_rf_cm.pdf'))

# save metrics
report=classification_report(y_test,y_pred_num,output_dict=True)
report=report['weighted avg']
report['average_precision']=average_precision
report['auc']=roc_auc

#pd.DataFrame([report]).to_csv("report.csv")
pd.DataFrame([report]).to_csv(os.path.join(results_dir,out_file+'_rf_metrics.csv'))


###############
# interpretation

imp_rf = pd.DataFrame({'feature': feature_names, 'importance': rf.feature_importances_})
imp_rf.sort_values('importance', inplace=True, ascending=False)
pd.DataFrame([imp_rf]).to_csv(os.path.join(results_dir,out_file+'_imp_rf.csv'))

coefs = pd.DataFrame({'feature': feature_names, 'coef': lr.coef_.ravel(), 'abs_coef': abs(lr.coef_).ravel()})
coefs.sort_values('abs_coef', ascending=False, inplace=True)
pd.DataFrame([coefs]).to_csv(os.path.join(results_dir,out_file+'_coefs_LR.csv'))

coefs[['feature', 'coef']]
