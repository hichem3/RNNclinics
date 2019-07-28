# processing text reports, splitting train-val-test and saving to disk
import logging, sys

import pandas as pd
import numpy as np

from clintk.utils.connection import get_engine, sql2df

from prescreen.vcare.reports import fetch_and_fold as rep_vc
from prescreen.simbad.parse_reports import fetch_and_fold as rep_sb
from prescreen.evaluation.fetch_reports import get_frames as ph12_sb

# set paths
# phase 1 from simbad and ventura care
# expected in path :  cr_sfditep_2012.xlsx, ditep_inclus.xlsx, ditep_sf.xlsx
table, path = 'event', '/home/v_charvet/workspace/data/'

# instantiation of logger
stdout = logging.StreamHandler(sys.stdout)
stdout.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger = logging.getLogger('default')
logger.setLevel(logging.INFO)
logger.addHandler(stdout)

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

df_rep_vc = rep_vc(table, engine, 'patient_target', 1)
df_rep_sb = rep_sb(path + '/cr_sfditep_2012.xlsx', engine, 'patient_target_simbad', 1)

df_rep_sb.rename({'id': 'patient_id'}, axis=1, inplace=True)

df_rep = pd.concat([df_rep_vc, df_rep_sb], ignore_index=True, sort=False)

df_rep = df_rep.merge(target[['id', 'screenfail']], left_on='patient_id', right_on='id')
logging.info("Displaying head of reports dataframe")

#####
# load Phase1/2 cohort from DITEP

df_ph12=ph12_sb(path)

df_all = pd.concat([df_rep, df_ph12], axis=0, ignore_index=True)

# assign Train, Cval, Test
df_all=df_all.assign(Cohort=np.random.choice(["Train","Val","Test"],
                                           df_all.shape[0], p=[0.7, 0.15, 0.15]))

logging.info("Loading and splitting done")
print(df_all.shape)
df_all.head()
pd.value_counts(df_all['Cohort'])

df_all=df_all.assign(CompleteValues=df_all['value'].apply(len)!=0)

#write file
df_all.to_csv(path + "/df_all.csv")
logging.info("Writing")

