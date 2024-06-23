#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

yamlPath="$1"

function parse_yaml {
   local prefix=$2
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
export WORKSPACE=/host

#Configuration passwordless ssh
bash ./host/script/start_ssh_service.sh root docker /host/ip_config.txt

export CONDA_PREFIX=/usr/local

echo -e "\nStarting distributed workflow..."

if [ "${distributed_build_graph}" = True ]; then
    if [[ ! -f "/DATA_IN/${env_in_data_filename}" ]]; then
        echo -e "\n/DATA_IN/${env_in_data_filename} does not exist"
    fi;
    echo -e "\nBuilding graph..."
    config="/CONFIGS/${env_tabular2graph_config_file}"
    bash ./host/script/run_build_graph.sh "/DATA_IN/${env_in_data_filename}" /GNN_TMP ${config} ${graph_CSVDataset_name}
fi;

if [ "${distributed_partition_graph}" = True ]; then
    echo -e "\nPartition graph..."
    part_path="/GNN_TMP/partitions"
    echo $part_path
    bash ./host/script/run_graph_partition.sh "/GNN_TMP/${graph_CSVDataset_name}" $distributed_num_parts $part_path
fi;

if [ "${distributed_gnn_training}" = True ]; then
    echo -e "\nStart GNN training..."
    part_path="/GNN_TMP/partitions"
    config_path="/CONFIGS/${env_train_config_file}"
    echo $config_path
    bash ./host/script/run_dist_train.sh "/DATA_IN/${env_in_data_filename}" "/GNN_TMP" "${part_path}" "${distributed_num_parts}" "/DATA_OUT" "${CONDA_PREFIX}" "${graph_name}" "${graph_CSVDataset_name}" "${config_path}" ${env_ssh_port}
fi;

if [ "${distributed_map_save}" = True ]; then
    echo "\nMapping to original graph IDs followed by mapping to CSV file output"
    echo "\nThis may take a while"
    part_path="/GNN_TMP/partitions"
    config="/CONFIGS/${env_tabular2graph_config_file}"
    echo ${config}
    bash ./host/script/run_map_save_dist.sh "/DATA_IN/${env_in_data_filename}" "${distributed_num_parts}" "${part_path}" "/DATA_OUT" "${config}"
fi;
