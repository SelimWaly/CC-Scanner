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

import time 
from xgboost_ray import RayDMatrix, RayParams, train, predict, RayFileType, RayShardingMode
import pandas as pd
import ray 
import numpy as np 
import shutil
from pathlib import Path 
import os 
import glob 
import math 
from sklearn import preprocessing
from category_encoders import TargetEncoder


edge_feature_data_path = '/workspace/data/graph/output/processed_data.csv'
edge_feature_ignore_cols = ['merchant_name','user', 'card', 'split']
save_train_path = '/workspace/data/graph/processed_data_split6/train'
save_valid_path = '/workspace/data/graph/processed_data_split6/valid'
save_test_path = '/workspace/data/graph/processed_data_split6/test'


df = pd.read_csv(edge_feature_data_path)
df.drop(columns=edge_feature_ignore_cols, inplace=True)

train_df = df[df["year"] < 2018]
valid_df = df[df["year"] == 2018]
test_df = df[df["year"] > 2018]
print(train_df.shape)
print(valid_df.shape)
print(test_df.shape)

def create_path(path):
    path = Path(path)

    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    
    os.makedirs(path)
    
    return None


def save_divide_df(df, save_format, save_data_path, num_partitions):
    create_path(save_data_path)
    
    if save_format == 'csv':
        df_splits = np.array_split(df, num_partitions)
        for i, data in enumerate(df_splits):
            data.to_csv(f"{save_data_path}/partition_{i}.csv", index=False)
    else:
        print("other data format not supported")
    print("done!")

save_divide_df(train_df, 'csv', save_train_path, 100)
save_divide_df(valid_df, 'csv', save_valid_path, 100)
save_divide_df(test_df, 'csv', save_test_path, 100)


ray.init(address="auto")
cpus_per_actor = 15
num_actors = 20
ray_params = RayParams(num_actors=num_actors, cpus_per_actor=cpus_per_actor, elastic_training=True, max_failed_actors=1, max_actor_restarts=2)


train_path = list(sorted(glob.glob(f'{save_train_path}/*.csv')))
valid_path = list(sorted(glob.glob(f'{save_valid_path}/*.csv')))
test_path = list(sorted(glob.glob(f'{save_test_path}/*.csv')))


dtrain = RayDMatrix(train_path,
                    label="is_fraud?", 
                    filetype=RayFileType.CSV)

dvalid = RayDMatrix(valid_path,
                    label="is_fraud?", 
                    filetype=RayFileType.CSV)

dtest = RayDMatrix(test_path,
                    label="is_fraud?", 
                    filetype=RayFileType.CSV)


model_params = {
          "learning_rate":0.1, 
          "eval_metric":"aucpr", 
          "objective":"binary:logistic"
         }
num_boost_round = 1000
eval_every = 100

tic = time.time()
model = train(model_params, 
                dtrain,
                evals=[(dtrain,'train'), (dvalid,'valid'), (dtest, 'test')],
                num_boost_round=num_boost_round,
                #maximize=True,
                verbose_eval=eval_every,
                ray_params=ray_params)
print("Training of baseline model with feature size 22 takes {:.0f} seconds".format(time.time() - tic))  

