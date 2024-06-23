#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

dataPath=$1
outPath=$2

#download dataset
#for now read from local file

#run edge featurization preprocessing
python ${WORKSPACE}/src/data_prep.py --raw_transaction_data_path ${dataPath} --edge_feature_data_path ${outPath}
