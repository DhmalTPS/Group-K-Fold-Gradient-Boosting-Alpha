import os, gc, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor, Pool

SEED = 36
N_FOLDS = 5
LGB_NUM_ROUNDS = 12000  
LGB_EARLY_STOP = 150*2
np.random.seed(SEED)

LGB_PARAMS = {
    'objective':         'regression',
    'metric':            'rmse',
    'learning_rate':     0.02,
    'num_leaves':        31,
    'max_depth':         -1,
    'min_child_samples': 300,       #2 Improvement 200 -> 300
    'feature_fraction':  0.5,
    'bagging_fraction':  0.7,
    'bagging_freq':      1,
    'lambda_l1':         1.0,       #1 No improvement  
    'lambda_l2':         8.0,       #1 Improvement 5.0 -> 8.0
    'min_gain_to_split': 0.01,
    'verbose':           -1,
    'n_jobs':            -1,
    'seed':              SEED,
}

XGB_PARAMS = {
    'objective':             'reg:squarederror',
    'n_estimators':          5000,
    'learning_rate':         0.02,
    'max_depth':             4,
    'min_child_weight':      200,
    'subsample':             0.7,
    'colsample_bytree':      0.5,
    'reg_alpha':             1.0,
    'reg_lambda':            5.0,
    'gamma':                 0.1,
    'tree_method':           'hist',
    'verbosity':             0,
    'n_jobs':                -1,
    'random_state':          SEED,
    'early_stopping_rounds': 150,
    }

CAT_PARAMS = dict(
    iterations=5000,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=10.0,
    min_data_in_leaf=200,
    random_seed=SEED,
    eval_metric='RMSE',
    od_type='Iter',
    od_wait=150,
    verbose=500,
    task_type='CPU',
    allow_writing_files=False,
)

train = pd.read_paraquet("./input/train-001.paraquet")
test = pd.read_paraquet("./input/test.paraquet")

train_ids = train['ID'].values.copy()
test_ids  = test['ID'].values.copy()
y_raw     = train['TARGET'].values.copy()

train.drop(columns=['ID','TARGET'], inplace=True)
test.drop(columns=['ID'], inplace=True)

# TARGET CLIP
clip_lo = y_raw.quantile(0.02)
clip_hi = y_raw.quantile(0.98)
y_series = y_raw.clip(clip_lo, clip_hi)
y = y_series.values.astype(np.float32)

all_cols =  train.columns.tolist()


