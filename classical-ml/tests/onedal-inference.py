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

import pandas as pd 
import time 
from xgboost import DMatrix, train
import daal4py as d4p
import pickle 
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
import xgboost as xgb
import sys 


export OMP_NUM_THREADS=24

if __name__ == "__main__":
    api = sys.argv[1]
    mode = sys.argv[2]

    edge_feature_data_path = '/workspace/data/graph/output/processed_data.csv'
    edge_feature_ignore_cols = ['merchant_name','user', 'card', 'split']

    # read data
    tic = time.time()
    df = pd.read_csv(edge_feature_data_path) 
    df.drop(columns=edge_feature_ignore_cols, inplace=True)
    print("time to read dataframe = {:.0f} seconds".format(time.time() - tic))

    # split data 
    train_df = df[df["year"] < 2018]
    val_df = df[df["year"] == 2018]
    test_df = df[df["year"] > 2018]
    print("data splitted!")

    if api == 'xgboost':
        model_params = {
          "learning_rate":0.1, 
          "eval_metric":"aucpr", 
          "objective":"binary:logistic"
         }
        
        num_boost_round = 100
        eval_every = 50

        Dtrain = DMatrix(train_df.drop(columns=["is_fraud?"]), label=train_df["is_fraud?"])
        Dval = DMatrix(val_df.drop(columns=["is_fraud?"]), label=val_df["is_fraud?"])

        model = train(model_params, 
              Dtrain, 
              num_boost_round=num_boost_round, 
              evals=[(Dtrain, 'train'), (Dval, 'val')],
              verbose_eval=eval_every)

    elif api == 'scikit':
        model_params = {
          "learning_rate":0.1, 
          "eval_metric":"aucpr", 
          "objective":"binary:logistic",
          "n_estimators": 100
         }
        eval_every = 50
        model = xgb.XGBClassifier(**model_params).fit(train_df.drop(columns=["is_fraud?"]),
                                                    train_df["is_fraud?"],
                                                    eval_set=[(val_df.drop(columns=["is_fraud?"]), val_df["is_fraud?"])],
                                                    verbose=eval_every)
    else:
        print("api is not correctly specified")

    if mode == 'native':

        tic = time.time()
        Dtest = DMatrix(test_df.drop(columns=["is_fraud?"]), label=test_df["is_fraud?"])
        predictions = model.predict(Dtest)
        print("native evaluation time = {:.0f} seconds".format(time.time() - tic))
        
    elif mode == 'onedal':

        daal_model = d4p.get_gbt_model_from_xgboost(model)
        tic = time.time()
        daal_predictions = d4p.gbt_classification_prediction(nClasses=2, resultsToEvaluate="computeClassLabels|computeClassProbabilities").compute(test_df.drop(columns=["is_fraud?"]), daal_model)
        print("onedal evaluation time = {:.0f} seconds".format(time.time() - tic))
        #daal_labels = daal_predictions.prediction[:, 0]
        predictions = daal_predictions.probabilities[:, 1]
    else:
        print("mode is not correctly specified.")

    print("computing the evalution metrics aucpr: ")
    precision, recall, _ = precision_recall_curve(test_df['is_fraud?'], predictions)
    print(auc(recall, precision))
