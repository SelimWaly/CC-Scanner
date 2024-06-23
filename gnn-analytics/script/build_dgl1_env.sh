#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

yes | conda create -n dgl1.0 python=3.8 -c anaconda

eval "$(conda shell.bash hook)"

conda activate dgl1.0

yes | conda install pip cmake
yes | conda install pytorch  cpuonly -c pytorch
yes | conda install -c intel intel-extension-for-pytorch
yes | conda install -c dglteam dgl
yes | conda install -c conda-forge psutil
yes | conda install -c conda-forge tqdm
yes | conda install -c conda-forge ogb
yes | conda install -c conda-forge numpy
yes | conda install -c conda-forge scikit-learn
yes | conda install -c conda-forge pydantic
yes | conda install -c conda-forge pyyaml
yes | conda install -c conda-forge chardet
yes | conda install -c conda-forge torchmetrics
yes | conda install -c conda-forge htop
yes | conda install conda-pack -c conda-forge
