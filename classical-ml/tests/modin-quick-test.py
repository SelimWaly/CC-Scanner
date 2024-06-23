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

import numpy as np
import sys
from scipy.special import expit
import time 
import math 

class TargetEncoder2:
    def __init__(self, input_col=None, target_col=None, min_samples_leaf=20, smoothing=10):
        self.input_col = input_col
        self.target_col = target_col
        self.min_samples_leaf = min_samples_leaf 
        self.smoothing = smoothing       
        self.mapping = None
        self._mean = None
        
    def fit(self, X):
        mapping = {}
        
        y = X[self.target_col]
        scalar = self._mean = y.mean()       
        
        if X[self.input_col].dtype.name == 'category':
            X[self.input_col] = X[self.input_col].cat.codes
        
        stats = y.to_frame().groupby(X[self.input_col]).agg({self.target_col:['count', 'mean']})
        stats.columns = stats.columns.droplevel(0)
        
        smoove = self._weighting(stats['count'])
        
        smoothing = scalar * (1-smoove) + stats['mean'] * smoove 
        
        smoothing.loc[-12] = scalar
        mapping[self.input_col] = smoothing 

        return mapping 
    
    def transform(self, X):
        
        if X[self.input_col].dtype.name == 'category':
            result = X[self.input_col].cat.codes.map(self.mapping[self.input_col])
        else:
            result = X[self.input_col].map(self.mapping[self.input_col])
        
        if result.isnull().sum() > 0:
            result.fillna(self.mapping[self.input_col].loc[-12], inplace=True)
        return result
    
    
    def fit_transform(self, X):
        
        _X = X.copy()
        self.mapping = self.fit(_X)
        
        return self.transform(X)
    
    def _weighting(self, n):
        tmp = (n - self.min_samples_leaf) / self.smoothing
        res = tmp.apply(lambda sr: expit(sr))
        return res
    
    
if __name__ == "__main__":

    mode = sys.argv[1]

    if mode == 'pandas':
        import pandas as pd 
        from category_encoders import TargetEncoder
    elif mode == 'modin':
        import modin.pandas as pd 
        import ray
        ray.init('auto', runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}}, log_to_driver=False)
    else:
        print("only pandas or modin is accepted")

    # the tabformer data has 24386900 transactions
    raw_transaction_data_path = "/workspace/data/graph/input/card_transaction.v1.csv"
    edge_feature_data_path='/workspace/data/graph/output/processed_data.csv'
    edge_feature_ignore_cols = ['merchant_name','user', 'card', 'split']

    # step 1 : read and clean the dataframe 
    tic = time.time()
    df = pd.read_csv(raw_transaction_data_path)
    print("Time of data reading = {} seconds".format(math.ceil(time.time() - tic)))


    tic = time.time()
    df.columns = df.columns.str.replace(' ','_').str.lower()
    print("Time of changing df column names = {} seconds".format(math.ceil(time.time() - tic)))

    tic = time.time()
    df["merchant_city"] = df["merchant_city"].astype('category')
    df["merchant_state"] = df["merchant_state"].astype('category')
    df["zip"] = df["zip"].astype('str').astype('category')
    df["mcc"] = df["mcc"].astype('category')
    print("Time of changing column data types = {} seconds".format(math.ceil(time.time() - tic)))

    tic = time.time()
    df['merchant_id'] = df['merchant_name'].astype('category').cat.codes
    df["is_fraud?"] = df["is_fraud?"].astype('category').cat.codes
    print("Time of categorify columns = {} seconds".format(math.ceil(time.time() - tic)))

    tic = time.time()
    df["amount"] = df["amount"].str.strip('$')
    df['time'] = df['time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    print("Time of modifying column values = {} seconds".format(math.ceil(time.time() - tic)))
    
    tic = time.time()
    df["card_id"] = df["user"].astype("str") + df["card"].astype("str")
    print("Time of adding new column from existing columns = {} seconds".format(math.ceil(time.time() - tic)))

    tic = time.time()
    df["amount"] = df["amount"].astype('float32')
    df['time'] = df['time'].astype('uint8')
    df["card_id"] = df["card_id"].astype('float32') 
    print("Time of changing column data types = {} seconds".format(math.ceil(time.time() - tic)))

    tic = time.time()
    df['time'] = (df['time'] - df['time'].min())/(df['time'].max() - df['time'].min()) 
    print("Time of modify column values = {} seconds".format(math.ceil(time.time() - tic)))

    tic = time.time()
    oneh_enc_cols = ["use_chip"]
    df = pd.concat([df, pd.get_dummies(df[oneh_enc_cols])], axis=1)
    df.drop(columns=['use_chip'],axis=1, inplace=True)  
    print("Time of one-hot encoding = {} seconds".format(math.ceil(time.time() - tic)))

    tic = time.time()
    if mode == 'pandas':
        exploded = df["errors?"].map(lambda x: str(x).split(',')).explode() 
        raw_one_hot = pd.get_dummies(exploded, columns=["errors?"])
        errs = raw_one_hot.groupby(raw_one_hot.index).sum()
        df = pd.concat([df, errs], axis=1)
        col_names = df.columns
        if '' in col_names or 'nan' in col_names: 
            df.drop(columns=['', 'nan'], axis=1, inplace=True)
        df.drop(columns=['errors?'],axis=1, inplace=True) 
    elif mode == 'modin':
        exploded = df["errors?"].map(lambda x: str(x).split(',')).explode().to_frame()
        raw_one_hot = pd.get_dummies(exploded, columns=["errors?"])
        errs = raw_one_hot.groupby(level=0).sum()
        df = pd.concat([df, errs], axis=1)
        df.columns = df.columns.str.replace('errors?'+'_', '')
        col_names = df.columns
        if '' in col_names or 'nan' in col_names: 
            df.drop(columns=['', 'nan'], axis=1, inplace=True)
        df.drop(columns=['errors?'],axis=1, inplace=True) 
    else:
        print("only pandas or modin is accepted")
    print("Time of multi-hot encoding = {} seconds".format(math.ceil(time.time() - tic)))

    tic = time.time()
    df["split"] = pd.Series(np.zeros(df.shape[0]), dtype=np.int8)
    print("Time of adding new column split = {} seconds".format(math.ceil(time.time() - tic)))

    tic = time.time()
    df.loc[df["year"] == 2018, "split"] = 1
    df.loc[df["year"] > 2018, "split"] = 2
    print("Time of modifying column split = {} seconds".format(math.ceil(time.time() - tic)))

    tic = time.time()
    train_card_ids = df.loc[df["split"] == 0, "card_id"]
    train_merch_ids = df.loc[df["split"] == 0, "merchant_id"]
    print("Time of defining new variables = {} seconds".format(math.ceil(time.time() - tic)))

    tic = time.time()
    df.loc[(df["split"] != 0) & ~df["card_id"].isin(train_card_ids), "split"] = 3
    df.loc[(df["split"] != 0) & ~df["merchant_id"].isin(train_merch_ids), "split"] = 3
    print("Time of modifying column split = {} seconds".format(math.ceil(time.time() - tic)))


    tic = time.time()
    train_df = df[df["split"] == 0]
    valtest_df = df[(df["split"] == 1) | (df["split"] == 2)] 
    print("Time of data splitting = {} seconds".format(math.ceil(time.time() - tic)))

    tic = time.time()
    if mode == 'pandas':

        high_card_cols = ["merchant_city", "merchant_state", "zip", "mcc"]
        for col in high_card_cols:
            tgt_encoder = TargetEncoder(smoothing=0.001)
            train_df[col] = tgt_encoder.fit_transform(train_df[col], train_df["is_fraud?"]).astype('float32')
            valtest_df[col] = tgt_encoder.transform(valtest_df[col]).astype('float32')

    elif mode == 'modin':

        high_card_cols = ["merchant_city", "merchant_state", "zip", "mcc"]
        for col in high_card_cols:
            tgt_encoder = TargetEncoder2(input_col=col, target_col="is_fraud?", smoothing=0.001)
            train_df[col] = tgt_encoder.fit_transform(train_df)
            valtest_df[col] = tgt_encoder.transform(valtest_df)
    else:
        print("only pandas or modin is accepted")
    print("Time of target encoding = {} seconds".format(math.ceil(time.time() - tic)))


    tic = time.time()
    df_merge = pd.concat([train_df, valtest_df])
    print("Time of merging data = {} seconds".format(math.ceil(time.time() - tic)))

    # step 5 : write edge features to a file
    tic = time.time()
    df_merge.to_csv(edge_feature_data_path, index=False)  
    print("Writing edge features to csv file takes {} seconds".format(math.ceil(time.time() - tic)))

    if mode == 'modin':
        ray.shutdown()