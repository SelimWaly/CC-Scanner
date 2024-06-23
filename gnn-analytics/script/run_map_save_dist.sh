#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

PROCESSED_DATA=$1
NUMPART=$2
PART_DIR=$3
OUT_DIR=$4
CONFIG=$5

CURR_PART_DIR=$PART_DIR/tabformer_${NUMPART}parts
OUT_DATA=${OUT_DIR}/tabular_with_gnn_emb.csv

python ${WORKSPACE}/src/map_emb.py --processed_data_path ${PROCESSED_DATA} --partition_data_path ${CURR_PART_DIR} --model_emb_path ${OUT_DIR} --out_data ${OUT_DATA} --tab2graph_cfg ${CONFIG}
