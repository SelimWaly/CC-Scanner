"""
Copyright [2022-23] [Intel Corporation]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#!/usr/bin/env python
# coding: utf-8
import os, time, gc, sys, glob
import pandas as pd
import numpy as np
import xgboost as xgb
import yaml
from pathlib import Path
import shutil
import os
import sys
from src.utils.compute_metrics import *
RECSYS_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, f"{RECSYS_PATH}")
from features import *


very_start = time.time()

LABEL_NAMES = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']

XGB_PARAMS = { 
    'max_depth':8, 
    'learning_rate':0.1, 
    'subsample':0.8,
    'colsample_bytree':0.8, 
    'eval_metric':'logloss',
    'objective':'binary:logistic',
    'tree_method':'hist',
    "random_state":42
}

feature_list_stage1 = []
feature_list_stage1.append(stage1_reply_features[:-1])
feature_list_stage1.append(stage1_retweet_features[:-1])
feature_list_stage1.append(stage1_comment_features[:-1])
feature_list_stage1.append(stage1_like_features[:-1])

feature_list_stage2 = []
feature_list_stage2.append(stage2_reply_features[:-1])
feature_list_stage2.append(stage2_retweet_features[:-1])
feature_list_stage2.append(stage2_comment_features[:-1])
feature_list_stage2.append(stage2_like_features[:-1])


def recsys_training(data_path, model_save_path, pred_save_path, stage):

    if stage == 'stage1':
        feature_list = feature_list_stage1
        train = pd.read_parquet(f'{data_path}/stage1_train/') 
        valid = pd.read_parquet(f'{data_path}/stage1_valid/')
        print(train.shape)
        print(valid.shape)
    elif stage == 'stage2':
        feature_list = feature_list_stage2 
        train = pd.read_parquet(f'{data_path}/stage2_train_pred.parquet')
        valid = pd.read_parquet(f'{data_path}/stage2_valid_pred.parquet')
        print(train.shape)
        print(valid.shape)
    else:
        raise ValueError("pls specify either stage1 or stage2 for stage value.")
    
    oof = np.zeros((len(valid),len(LABEL_NAMES)))

    for numlabel in range(4):
        name = LABEL_NAMES[numlabel]
        print('#'*25);print('###',name);print('#'*25)

        print("Training.....")
        dtrain = xgb.DMatrix(data=train[feature_list[numlabel]], label=train[name])
        dvalid = xgb.DMatrix(data=valid[feature_list[numlabel]], label=valid[name])

        model = xgb.train(XGB_PARAMS, 
                dtrain=dtrain,
                evals=[(dtrain,'train'),(dvalid,'valid')],
                num_boost_round=250,
                early_stopping_rounds=25,
                #maximize=True,
                verbose_eval=25)    
        model.save_model(f"{model_save_path}/xgboost_{name}_{stage}.model")

        print('Predicting...')
        oof[:,numlabel] = model.predict(dvalid)

    if stage == 'stage1':
        for i in range(4):
            valid[f"pred_{LABEL_NAMES[i]}"] = oof[:,i]
        valid[["tweet_id","engaging_user_id",f"pred_{LABEL_NAMES[0]}",f"pred_{LABEL_NAMES[1]}",f"pred_{LABEL_NAMES[2]}",f"pred_{LABEL_NAMES[3]}"]].to_csv(f"{pred_save_path}/xgboost_pred_{stage}.csv",index=0)

    txts = ''
    sumap = 0
    sumrce = 0
    for i in range(4):
        ap = compute_AP(oof[:,i],valid[LABEL_NAMES[i]].values)
        rce = compute_rce_fast(oof[:,i],valid[LABEL_NAMES[i]].values)
        txt = f"{LABEL_NAMES[i]:20} AP:{ap:.5f} RCE:{rce:.5f}"
        print(txt)

        txts += "%.4f" % ap + ' '
        txts += "%.4f" % rce + ' '
        sumap += ap
        sumrce += rce
    print(txts)
    print("AVG AP: ", sumap/4.)
    print("AVG RCE: ", sumrce/4.)


def data_merge(data_path, pred_save_path):

    df1 = pd.read_parquet(f"{data_path}/stage2_train/")
    df2 = pd.read_parquet(f"{data_path}/stage2_valid/")

    pred_path = f"{pred_save_path}/xgboost_pred_stage1.csv"
    preds = pd.read_csv(pred_path)

    index_cols = ['tweet_id', 'engaging_user_id']
    df1 = df1.merge(preds, on=index_cols, how="left")
    df2 = df2.merge(preds, on=index_cols, how="left")

    df1.to_parquet(f"{data_path}/stage2_train_pred.parquet")
    df2.to_parquet(f"{data_path}/stage2_valid_pred.parquet")

    print("data merged!")

if __name__ == "__main__":

    data_path = '/workspace/data/processed'
    model_save_path = '/workspace/tmp/models'
    pred_save_path = '/workspace/tmp/data'

    recsys_training(data_path, model_save_path, pred_save_path, 'stage1')
    data_merge(data_path, pred_save_path)
    recsys_training(data_path, model_save_path, pred_save_path, 'stage2')
    
    print('This script took %.1f seconds'%(time.time()-very_start))