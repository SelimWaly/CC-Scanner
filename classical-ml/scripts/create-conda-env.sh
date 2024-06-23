#!/bin/bash

SCRIPTS_DIR=$(cd $(dirname $0); pwd)
conda create -n dcml python=3.10 -c anaconda -y

eval "$(conda shell.bash hook)"

conda activate dcml

pip install -r $SCRIPTS_DIR/../requirements.txt
