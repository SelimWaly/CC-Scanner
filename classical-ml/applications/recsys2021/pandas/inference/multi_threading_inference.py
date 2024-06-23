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

import os, time, gc, sys, glob
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from features import *
from multiprocessing import Pool
import yaml
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
very_start = time.time()

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..','..','..'))

try: 
    with open(os.path.join(ROOT_DIR,'config.yaml'),'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print("Errors reading the config file.")


data_path = config['files']['data_path']
model_save_path = config['files']['model_save_path'] 
pred_save_path = config['files']['pred_save_path'] 
DEBUG = config['training']['debug']

if DEBUG:
    model_save_path = Path(os.path.join(model_save_path, 'test'))
    pred_save_path = Path(os.path.join(pred_save_path, 'test'))

stage1_thread = 28
stage2_thread = 40

if __name__ == "__main__":
    ######## Load data
    t1 = time.time()
    #test = pd.read_parquet(f'{data_path}/test')  
    test = pd.read_parquet(f'{data_path}/stage12_test_0.parquet')  
    if DEBUG:
        test = test[:10000]
    print(test.shape)
    print(f"load data took {time.time() - t1} s")

    ######## Feature list for each target
    label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    
    stage1_feature_list = {}
    stage1_feature_list["reply_timestamp"] = stage1_reply_features
    stage1_feature_list["retweet_timestamp"] = stage1_retweet_features
    stage1_feature_list["retweet_with_comment_timestamp"] = stage1_comment_features
    stage1_feature_list["like_timestamp"] = stage1_like_features

    stage2_feature_list = {}
    stage2_feature_list["reply_timestamp"] = stage2_reply_features
    stage2_feature_list["retweet_timestamp"] = stage2_retweet_features
    stage2_feature_list["retweet_with_comment_timestamp"] = stage2_comment_features
    stage2_feature_list["like_timestamp"] = stage2_like_features

    ######## Stage1 prediction
    t1 = time.time()
    # Split data into multi-instance
    indexs = [i for i in range(stage1_thread)]
    step = int(len(test)/stage1_thread)
    tests = []
    for i in range(stage1_thread):
        if i<stage1_thread-1:
            tests.append(test[i*step:(i+1)*step])
        else:
            tests.append(test[i*step:])

    # Predict with multiprocessing
    def prediction1(index):
        start = time.time()
        test1 = tests[index]
        oof = np.zeros((len(test1),len(label_names)))

        for numlabel in range(4):
            name = label_names[numlabel]
            dtest = xgb.DMatrix(data=test1[stage1_feature_list[name]], label=test1[name])

            model = xgb.Booster(model_file=f"{model_save_path}/xgboost_{name}_stage1.model")
            oof[:, numlabel] += model.predict(dtest)
        print(f"{index} predict took {time.time() - start} s")
        return oof

    pool = Pool(stage1_thread)
    preds1 = pool.map(prediction1,indexs)
    pool.close()
    pool.join()
    gc.collect()
    
    # Merge result
    predss1 = np.concatenate(preds1)
    for i in range(4):
        test[f"pred_{label_names[i]}"] = predss1[:,i]
    print(test.shape)
    print(f"stage1 took {time.time() -t1} s")

    ######## Stage2 prediction
    t1 = time.time()
    # Split data into multi-instance
    indexs = [i for i in range(stage2_thread)]
    step = int(len(test)/stage2_thread)
    tests = []
    for i in range(stage2_thread):
        if i<stage2_thread-1:
            tests.append(test[i*step:(i+1)*step])
        else:
            tests.append(test[i*step:])

    # Predict with multiprocessing
    def prediction2(index):
        start = time.time()
        test1 = tests[index]
        oof = np.zeros((len(test1),len(label_names)))

        for numlabel in range(4):
            name = label_names[numlabel]
            dtest = xgb.DMatrix(data=test1[stage2_feature_list[name]], label=test1[name])

            model = xgb.Booster(model_file=f"{model_save_path}/xgboost_{name}_stage2.model")
            oof[:, numlabel] += model.predict(dtest)
        print(f"{index} predict took {time.time() - start} s")
        return oof

    pool = Pool(stage2_thread)
    preds2 = pool.map(prediction2,indexs)
    pool.close()
    pool.join()
    gc.collect()

    # Merge result
    predss2 = np.concatenate(preds2)
    for i in range(4):
        test[f"pred2_{label_names[i]}"] = predss2[:,i]
    print(test.shape)
    print(f"stage2 took {time.time() -t1} s")

    ######## Save to csv
    t1 = time.time()
    test[['engaging_user_id', 'tweet_id', f"pred2_{label_names[0]}",f"pred2_{label_names[1]}",f"pred2_{label_names[2]}",f"pred2_{label_names[3]}"]].to_csv(f"{pred_save_path}/xgboost_inference.csv",index=0)
    print(f"save took {time.time() -t1} s")

    print(f"totally took {time.time() -very_start} s")
