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
from xgboost_ray import RayDMatrix, RayParams, train, predict, RayFileType
from sklearn.metrics import log_loss, average_precision_score
import yaml
from pathlib import Path
import ray 
from ray import tune 
from ray.tune.suggest.optuna import OptunaSearch

RECSYS_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, f"{RECSYS_PATH}")
from features import *

very_start = time.time()
ray.init(address='auto')
cpus_per_actor = 15
num_actors = 20
num_samples = 50
ray_params = RayParams(num_actors=num_actors, cpus_per_actor=cpus_per_actor, elastic_training=True, max_failed_actors=1, max_actor_restarts=1)

space = {
    'objective':'binary:logistic',
    'tree_method':'hist',
    'eval_metric':"logloss",
    'random_state':42,
    "max_depth":6,
    #"max_depth": tune.randint(1, 9),
    "subsample":0.95,
    "colsample_bytree":0.75,
    "eta":0.1,
    "gamma":0,
    "min_child_weight":0.706382319,
    "scale_pos_weight": tune.loguniform(0.5, 2),
    "max_delta_step":9,
    'reg_alpha':tune.uniform(0,10),
    'reg_lambda':tune.uniform(0,2)
    #"min_child_weight": tune.choice([1, 2, 3]),
    # "eta": tune.loguniform(1e-4, 1e-1),
    # "subsample": tune.uniform(0.5, 1.0),
    # "colsample_bytree": tune.uniform(0.5, 1.0)
}

optuna_search = OptunaSearch(metric="eval-logloss", mode="min")

very_start = time.time()

def compute_AP(pred, gt):
    return average_precision_score(gt, pred)

def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr

def compute_rce_fast(pred, gt):
    cross_entropy = log_loss(gt, pred)
    yt = np.mean(gt)     
    strawman_cross_entropy = -(yt*np.log(yt) + (1 - yt)*np.log(1 - yt))
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0


best_configs = {}
 
if __name__ == "__main__":
    ######## Load data
    save_data_path = '/workspace/data/processed'
    model_save_path = '/workspace/tmp/models'
    pred_save_path = '/workspace/tmp/data'

    train_path = list(sorted(glob.glob(f'{save_data_path}/stage2_train/*.parquet')))
    valid_path = list(sorted(glob.glob(f'{save_data_path}/stage2_valid/*.parquet')))
    
    valid = pd.read_parquet(f'{save_data_path}/stage2_valid/')    

    print(valid.shape)

    ######## Feature list for each target
    label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    feature_list = []
    feature_list.append(stage2_reply_features)
    feature_list.append(stage2_retweet_features)
    feature_list.append(stage2_comment_features)
    feature_list.append(stage2_like_features)
    for i in range(1,2):
        print(len(feature_list[i])-1)

    oof = np.zeros((len(valid),len(label_names)))
    for numlabel in range(1,2):
        start = time.time()
        name = label_names[numlabel]
        print('#'*25);print('###',name);print('#'*25)

        def train_model(config, ray_params):
            dtrain = RayDMatrix(
                            train_path,
                            label=name,  # Will select this column as the label
                            columns=feature_list[numlabel],
                            # ignore=["total_amount"],  # Optional list of columns to ignore
                            filetype=RayFileType.PARQUET)
        
            dvalid = RayDMatrix(
                    valid_path,
                    label=name,  # Will select this column as the label
                    columns=feature_list[numlabel],
                    # ignore=["total_amount"],  # Optional list of columns to ignore
                    filetype=RayFileType.PARQUET)

            evals_result = {}
            bst = train(
                params=config,
                dtrain=dtrain,
                evals_result=evals_result,
                evals=[(dtrain,'train'), (dvalid, 'eval')],
                verbose_eval=False,
                num_boost_round=3000,
                early_stopping_rounds=25,
                ray_params=ray_params)

            #bst.save_model("tuned.xgb")
            #print("Final validation error: {:.4f}".format(evals_result["eval"]["error"][-1]))

     
        analysis = tune.run(
                tune.with_parameters(train_model, ray_params=ray_params),
                config=space,
                search_alg=optuna_search,
                num_samples=num_samples,
                metric="eval-logloss", 
                mode="min",
                local_dir = '/workspace/tmp/ray',
                resources_per_trial=ray_params.get_tune_resources())

        #accuracy = 1. - analysis.best_result["eval-error"]
        #best_configs.append(analysis.best_config)
        print(f"Best model parameters: {analysis.best_config}")
        #print(f"Best model total accuracy: {accuracy:.4f}")

  
    print('This notebook took %.1f seconds'%(time.time()-very_start))