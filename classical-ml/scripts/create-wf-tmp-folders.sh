#!/bin/bash

tmp_folder="$1"

echo -e "\ngenerate workflow tmp folders...."
mkdir -p $tmp_folder/hdfs/dn 
mkdir -p $tmp_folder/hdfs/nn
mkdir -p $tmp_folder/hdfs/tmp 
mkdir -p $tmp_folder/spark
mkdir -p $tmp_folder/ray
mkdir -p $tmp_folder/models
mkdir -p $tmp_folder/data
mkdir -p $tmp_folder/logs

chmod -R 777 $tmp_folder
