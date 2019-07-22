# processing text reports, splitting train-test and saving to disk
import logging, sys

import pandas as pd
import numpy as np
import seaborn as sns

from clintk.utils.connection import get_engine, sql2df

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV, train_test_split
from prescreen.vcare.reports import fetch_and_fold as rep_vc
from prescreen.simbad.parse_reports import fetch_and_fold as rep_sb

# instantiation of logger
stdout = logging.StreamHandler(sys.stdout)
stdout.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger = logging.getLogger('default')
logger.setLevel(logging.INFO)
logger.addHandler(stdout)

# constant variables
#TEST_SPLIT = ..

## Fetching data
logger.info("Loading data from Simbad and VCare")

engine = get_engine('vcare', 'srt-ap-92', 'vcare')

target_vcare = sql2df(engine, 'patient_target')
target_simbad = sql2df(engine, 'patient_target_simbad')

target_vcare = target_vcare.loc[:, ['nip', 'id', 'screenfail']]
target_simbad.drop(['prescreen', 'index'], axis=1, inplace=True)

target = pd.concat([target_vcare, target_simbad], axis=0, ignore_index=True)
logger.info("Target shape and head:")
print(target.shape)
target.head()

table, path = 'event', '../data/cr_sfditep_2012.xlsx'

df_rep_vc = rep_vc(table, engine, 'patient_target', 1)
df_rep_sb = rep_sb(path, engine, 'patient_target_simbad', 1)

df_rep_sb.rename({'id': 'patient_id'}, axis=1, inplace=True)

df_rep = pd.concat([df_rep_vc, df_rep_sb], ignore_index=True, sort=False)

df_rep = df_rep.merge(target[['id', 'screenfail']], left_on='patient_id', right_on='id')
logging.info("Displaying head of reports dataframe")
df_rep.head()

#x_rep_raw, y_rep = df_rep['value'], df_rep['screenfail']

# concatenate
#df_all=np.concatenate((df_rep['value'], df_rep['screenfail']),axis=1)

# splitting data for train/test
#X_train, X_test, y_train, y_test = train_test_split(x_rep_raw, y_rep, test_size=TEST_SPLIT)

# assign Train, Cval, Test
df_rep=df_rep.assign(Cohort=np.random.choice(["Train","Val","Test"], df_rep.shape[0], p=[0.7, 0.15, 0.15]))

logging.info("Loading and splitting done")
print(df_rep.shape)
df_rep.head()

#write file
df_rep.to_csv("df_rep.csv")
logging.info("Writing")
