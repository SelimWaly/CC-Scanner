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

echo -e "\nStarting single node workflow..."
if [ "$single_build_graph" = True ]; then
    echo -e "\nBuilding graph..."
    graph_config="/CONFIGS/${env_tabular2graph_config_file}"
    bash ./host/script/run_build_graph.sh "/DATA_IN/${env_in_data_filename}" /GNN_TMP ${graph_config} ${graph_CSVDataset_name}
fi;
if [ "${single_gnn_training}" = True ]; then
    echo -e "\nStart GNN training..."
    config_path="/CONFIGS/${env_train_config_file}"
    echo $config_path
    bash ./host/script/run_train_single.sh "/DATA_IN/${env_in_data_filename}" "/GNN_TMP" "/DATA_OUT" ${graph_CSVDataset_name} "${config_path}"
fi;
if [ "${single_map_save}" = True ]; then
    echo "Mapping to original graph IDs followed by mapping to CSV file output"
    echo "This may take a while"
    graph_config="/CONFIGS/${env_tabular2graph_config_file}"
    bash ./host/script/run_map_save.sh "/DATA_IN/${env_in_data_filename}" "/DATA_OUT" "${graph_config}"
fi;
