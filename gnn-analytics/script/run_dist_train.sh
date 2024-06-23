#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

PROCESSED_DATA=$1
GNN_TMP=$2
PART_DIR=$3
NUMPART=$4
OUT_DIR=$5
CONDA_ENV=$6
GRAPH_NAME=$7
CSVDATASET=$8
yamlPath=$9
SSH_PORT=${10}


function parse_yaml {
   local prefix=${10}
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|,$s\]$s\$|]|" \
        -e ":1;s|^\($s\)\($w\)$s:$s\[$s\(.*\)$s,$s\(.*\)$s\]|\1\2: [\3]\n\1  - \4|;t1" \
        -e "s|^\($s\)\($w\)$s:$s\[$s\(.*\)$s\]|\1\2:\n\1  - \3|;p" $1 | \
   sed -ne "s|,$s}$s\$|}|" \
        -e ":1;s|^\($s\)-$s{$s\(.*\)$s,$s\($w\)$s:$s\(.*\)$s}|\1- {\2}\n\1  \3: \4|;t1" \
        -e    "s|^\($s\)-$s{$s\(.*\)$s}|\1-\n\1  \2|;p" | \
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)-$s[\"']\(.*\)[\"']$s\$|\1$fs$fs\2|p" \
        -e "s|^\($s\)-$s\(.*\)$s\$|\1$fs$fs\2|p" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" | \
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]; idx[i]=0}}
      if(length($2)== 0){  vname[indent]= ++idx[indent] };
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) { vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, vname[indent], $3);
      }
   }'
}

eval $(parse_yaml $yamlPath)

if [ "${CONDA_ENV}" != "/usr/local" ]; then
   eval "$(conda shell.bash hook)"
   conda activate dgl1.0
fi;
python --version

PY_EXEC=${CONDA_ENV}/bin/python3
EXEC_SCRIPT=${WORKSPACE}/src/train_dist_unsupervised_transductive_fraud.py

#tabformer CSVDataset
CSVDATASET=${GNN_TMP}/${CSVDATASET}
if [[ ! -d ${CSVDATASET} ]]; then
    echo -e "\n${CSVDATASET} does not exist. Need to build graph"
fi;
#model, embb and final csv outputs
mkdir -p ${OUT_DIR}
MODEL_OUT=${OUT_DIR}/model_graphsage_2L_64.pt
NEMB_OUT=${OUT_DIR}/node_emb.pt #these are the default names
NEMB_OUT_MAPPED=${OUT_DIR}/node_emb_mapped.pt #these are the default names
OUT_DATA=${OUT_DIR}/tabular_with_gnn_emb.csv

#GRAPH_NAME=( "tabformer_full_homo" )

#partition data directory
#NUMPART=2
CURR_PART_DIR=$PART_DIR/tabformer_${NUMPART}parts
PART_CONFIG=$CURR_PART_DIR/${GRAPH_NAME}.json

if [[ ! -d ${CURR_PART_DIR} ]]; then
    echo -e "\n${CURR_PART_DIR} does not exist. Need to partition graph"
fi;
#ip_config path
IP_CONFIG=${WORKSPACE}/ip_config.txt

# Folder and filename where you want your logs.
logdir=${WORKSPACE}/logs_dist
mkdir -p $logdir
logname=log_${GRAPH_NAME}_${NUMPART}n_$RANDOM
echo $logname
#set -x

cfg="$env_config_path/$env_train_config_file"
eval $(parse_yaml $yamlPath)
# minibatch size on each host
MB_SIZE=$workflow_spec_dataloader_params_batch_size
MB_SIZE_EVAL=$workflow_spec_dataloader_params_batch_size_eval

# hidden feature size
HIDDEN_FS=$workflow_spec_model_params_hidden_size

#number of layers in GNN encoder
N_LAYERS=$workflow_spec_model_params_num_layers

#Learning rate
L_RATE=$workflow_spec_model_params_learning_rate

# fanout per layer
FANOUT=$workflow_spec_sampler_params_fan_out

# num epochs to run for
EPOCHS=$workflow_spec_training_params_num_epochs

EVAL_EVERY=$workflow_spec_training_params_eval_every

#DGL distributed configuration
NTRAINER="${workflow_spec_dgl_params_num_trainers}"
NSAMPLER="${workflow_spec_dgl_params_num_samplers}"
NSERVER="${workflow_spec_dgl_params_num_servers}"

#setting OMP_NUM_THREADS to number of physical cores in one socket divided by the number of sampler processes
NUM_CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" |grep -Eo '[0-9]{1,3}'`
NUM_THREADS=$((NUM_CORES_PER_SOCKET/NSAMPLER))
echo -e "\nSetting OMP_NUM_THREADS=$NUM_THREADS"

echo ${WORKSPACE}
#using numactl to bind to single socket
python -u ${WORKSPACE}/src/launch.py --num_omp_threads $NUM_THREADS --workspace ${WORKSPACE} --num_trainers ${NTRAINER} --num_samplers ${NSAMPLER} --num_servers ${NSERVER} --part_config ${PART_CONFIG} --ip_config ${IP_CONFIG} --ssh_port ${SSH_PORT} "numactl -N 0 ${PY_EXEC} ${EXEC_SCRIPT} --graph_name ${GRAPH_NAME} --ip_config ${IP_CONFIG} --part_config ${PART_CONFIG} --num_epoch ${EPOCHS}  --num_hidden ${HIDDEN_FS} --num_layers ${N_LAYERS} --lr ${L_RATE} --fan_out ${FANOUT} --batch_size ${MB_SIZE} --batch_size_eval ${MB_SIZE_EVAL} --eval_every ${EVAL_EVERY} --CSVDataset_dir ${CSVDATASET} --remove_edge --model_out ${MODEL_OUT} --nemb_out ${NEMB_OUT}" |& tee ${logdir}/${logname}.txt
