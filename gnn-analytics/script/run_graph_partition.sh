#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

CSVDataset=$1
num_parts=$2
partPath=$3

if [[ ! -d "${CSVDataset}" ]]; then
    echo -e "\n${CSVDataset} does not exist. Need to build graph before you can partition it"
fi;
python ${WORKSPACE}/src/partition_tabformer_homo.py --CSVDataset ${CSVDataset} --num_part ${num_parts} --partition_out ${partPath}
