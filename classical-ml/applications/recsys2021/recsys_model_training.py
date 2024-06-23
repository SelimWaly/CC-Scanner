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
from features import *
import pyspark
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType, BooleanType

import os, sys
import time 
import argparse
import yaml 
import findspark
findspark.init()

from src.training.spark.model_training import XGBoostSparkClassifier
from src.utils.compute_metrics import *
from src.utils.data_utils import read_parquet_spark

very_start = time.time()

LABEL_NAMES = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']

XGB_PARAMS = dict(
    eta=0.1,
    maxDepth=8,
    evalMetric='logloss',
    objective='binary:logistic',
    treeMethod='hist',
    missing=0.0,
    seed=42,
    subsample=0.8,
    colsampleBytree=0.8,
    nthread=8,
    numWorkers=16,
    numEarlyStoppingRounds=25,
    numRound=250,
    maximizeEvaluationMetrics=False
)

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


def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()
    return udf(to_array_, ArrayType(DoubleType())).asNondeterministic()(col)

def setup_local():

    spark = SparkSession.builder.master('local[*]')\
        .appName("Recsys2021_xgboost_train")\
        .config("spark.driver.memory", '300g')\
        .config("spark.local.dir", "/workspace/tmp/spark")\
        .config("spark.task.cpus", '8')\
        .config("spark.sql.debug.maxToStringFields", 10000)\
        .getOrCreate()
    
    return spark

def setup_standalone(master_name):

    spark = SparkSession.builder.master(f'spark://{master_name}:7077')\
        .appName("Recsys2021_xgboost_train")\
        .config("spark.driver.memory", '25g')\
        .config("spark.local.dir", "/workspace/tmp/spark")\
        .config("spark.executor.memory", "30g")\
        .config("spark.executor.memoryOverhead", "15g")\
        .config("spark.executor.cores", "8")\
        .config("spark.executor.instances","16")\
        .config("spark.task.cpus","8")\
        .config("spark.driver.maxResultSize", "10g")\
        .getOrCreate()
    
    return spark

def train_recsys_models(spark, path_prefix, xgb_params, stage):
    sumap = 0
    sumrce = 0

    if stage == 'stage1':
        feature_list = feature_list_stage1
        train_data = read_parquet_spark(spark, path_prefix +'/recsys2021/datapre_stage1/stage1_train')
        valid_data = read_parquet_spark(spark, path_prefix +'/recsys2021/datapre_stage1/stage1_valid')
    elif stage == 'stage2':
        feature_list = feature_list_stage2
        train_data = read_parquet_spark(spark, path_prefix +'/recsys2021/stage2_train_pred.parquet')
        valid_data = read_parquet_spark(spark, path_prefix +'/recsys2021/stage2_valid_pred.parquet')
    else:
        raise ValueError("the value of stage is either stage1 or stage2. Pls specify the correct name")

    if stage == 'stage1':
        preds = valid_data.select(col("tweet_id"), col("engaging_user_id"))

    for i in range(4):
        name = LABEL_NAMES[i]
        print('#'*25);print('###',name);print('#'*25)

        vector_assembler = VectorAssembler()\
                            .setInputCols(feature_list[i])\
                            .setOutputCol("features")

        train_data_trans = vector_assembler.setHandleInvalid("keep").transform(train_data)
        valid_data_trans = vector_assembler.setHandleInvalid("keep").transform(valid_data)

        xgb_classifier = XGBoostSparkClassifier(xgb_params, name)
        xgb_clf_model = xgb_classifier.fit(train_data_trans)

        predictions = xgb_clf_model.transform(valid_data_trans)

        eval_results = predictions.withColumn("prob", to_array(col("probability")))\
                                    .select(col("tweet_id"), col("engaging_user_id"), col(name), col("prob")[1])\
                                    .withColumnRenamed("prob[1]", f"pred_{name}")
        
        if stage == 'stage1':
            preds = preds.join(eval_results.select(col("tweet_id"), col("engaging_user_id"),col(f"pred_{name}")), on=['tweet_id','engaging_user_id'], how='left')

        results = eval_results.select(col(name), col(f"pred_{name}")).toPandas()

        gt = results[name]
        pred = results[f"pred_{name}"]
        
        ap = compute_AP(pred, gt)
        rce = compute_rce_fast(pred, gt)
        
        sumap += ap
        sumrce += rce

        txt = f"{name:20} AP:{ap:.5f} RCE:{rce:.5f}"
        print(txt)

        print("saving models...")
        xgb_clf_model.write().overwrite().save(path_prefix+f"/recsys2021/models/xgboost_{name}_{stage}.model")

    print('-----------------------------------')
    print("AVG AP: ", sumap/4.)
    print("AVG RCE: ", sumrce/4.)

    if stage == 'stage1':
        preds.repartition(1).write.option("header",True).mode("overwrite").csv(path_prefix + f"/recsys2021/results/xgboost_pred_{stage}.csv")
    

def data_merge(spark, path_prefix):
    train_data = read_parquet_spark(spark, path_prefix+'/recsys2021/datapre_stage2/stage2_train')
    valid_data = read_parquet_spark(spark, path_prefix+'/recsys2021/datapre_stage2/stage2_valid')

    index_cols = ['tweet_id', 'engaging_user_id']
    pred = spark.read.options(header='True', inferSchema='True').csv(path_prefix + f"/recsys2021/results/xgboost_pred_stage1.csv")

    train_data = train_data.join(pred, on=index_cols, how='left')
    valid_data = valid_data.join(pred, on=index_cols, how='left')
    cols = train_data.columns
    for col_name in cols:
        if train_data.schema[col_name].dataType == BooleanType():
            train_data = train_data.withColumn(col_name, col(col_name).cast('integer'))
            valid_data = valid_data.withColumn(col_name, col(col_name).cast('integer'))

    train_data.repartition(1).write.mode("overwrite").parquet(path_prefix + "/recsys2021/stage2_train_pred.parquet")
    valid_data.repartition(1).write.mode("overwrite").parquet(path_prefix + "/recsys2021/stage2_valid_pred.parquet")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--config-file",
            required=True,
            type=str,
            help="speficy the config path")
    args, _ = parser.parse_known_args()

    try: 
        with open(args.config_file,'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print("Errors reading the config file.")

    num_node = config['env']['num_node']
    is_local = True if num_node == 1 else False
    master = config['env']['node_ips'][0]
    path_prefix = f"hdfs://{master}:9000/"

    if is_local:
        print("setting up spark local mode...")
        spark = setup_local()
    else:
        print("setting up spark standalone mode...")
        spark = setup_standalone(master)
    
    train_recsys_models(spark, path_prefix, XGB_PARAMS, 'stage1')
    
    data_merge(spark, path_prefix)

    train_recsys_models(spark, path_prefix, XGB_PARAMS, 'stage2')

    print('This script took %.1f seconds'%(time.time()-very_start))

    
    

    

    