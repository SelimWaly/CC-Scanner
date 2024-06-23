#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

start_time=$(date +%s)

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

yamlFileName=$(basename $yamlPath)
yamlAbsolutePath=$(readlink -f $yamlPath)
repoPath=$(pwd)
repoName=$(basename $(pwd))
hostName=$(hostname -I | awk '{print $1}')
currTime=$(date +%Y-%m-%d-%H-%M)
wfTmpFolder="$env_tmp_path/wf-session-${currTime}"

export WORKSPACE=$repoPath

#TODO: if preprocesssed data doesnt exist - run local utility
ENV_NAME="dgl1.0"

if [ "$env_bare_metal" = True ] ; then
    ## Prepare Conda Environment
    eval "$(conda shell.bash hook)"
    if conda env list | grep ${ENV_NAME}; then
        echo -e "\ndgl1.0 conda env already exists, activating environment"
        conda activate ${ENV_NAME}
    else
        echo -e "\nBuilding conda environment..."
        bash ./script/build_dgl1_env.sh
        conda activate ${ENV_NAME}
    fi;
    ## Single node bare metal steps
    if ((${env_num_node} == 1 )); then
        echo -e "\nStarting single node workflow..."
        if [ "${single_build_graph}" = True ]; then
            echo -e "\nBuilding graph..."
            config="${env_config_path}/${env_tabular2graph_config_file}"
            echo ${config}
            bash ./script/run_build_graph.sh "${env_data_path}/${env_in_data_filename}" ${env_tmp_path} "${config}" ${graph_CSVDataset_name}
        fi;

        if [ "${single_gnn_training}" = True ]; then
            echo -e "\nStart GNN training..."
            config_path="${env_config_path}/${env_train_config_file}"
            echo ${config_path}
            bash ./script/run_train_single.sh "${env_data_path}/${env_in_data_filename}" "${env_tmp_path}" "${env_out_path}" ${graph_CSVDataset_name} "${config_path}"
        fi;

        if [ "${single_map_save}" = True ]; then
            echo "\nMapping to original graph IDs followed by mapping to CSV file output"
            echo "\nThis may take a while"
            config="${env_config_path}/${env_tabular2graph_config_file}"
            echo ${config}
            bash ./script/run_map_save.sh "${env_data_path}/${env_in_data_filename}" "${env_out_path}" "${config}"
        fi;

    ## multi node bare metal steps
    elif ((${env_num_node} > 1 )); then
        echo -e "\nStarting distributed workflow..."

        echo -e "\nCreate ip_config.txt"
        rm ip_config.txt
        for ((i=1; i<=$env_num_node; i++)); do
        ip="env_node_ips_$i"
        echo ${!ip} >> ip_config.txt
        done

        if [ "${distributed_build_graph}" = True ]; then
            if [[ ! -f "${env_data_path}/${env_in_data_filename}" ]]; then
                echo -e "\n${env_data_path}/${env_in_data_filename} does not exist"
            fi;
            echo -e "\nBuilding graph..."
            config="${env_config_path}/${env_tabular2graph_config_file}"
            bash ./script/run_build_graph.sh "${env_data_path}/${env_in_data_filename}" ${env_tmp_path} ${config} ${graph_CSVDataset_name}
        fi;

        if [ "${distributed_partition_graph}" = True ]; then
            echo -e "\nPartition graph..."
            part_path="${env_tmp_path}/partitions"
            echo $part_path
            bash ./script/run_graph_partition.sh "${env_tmp_path}/${graph_CSVDataset_name}" $distributed_num_parts $part_path
        fi;

        if [ "${distributed_gnn_training}" = True ]; then
            echo -e "\nStart GNN training..."
            part_path="${env_tmp_path}/partitions"
            config_path="${env_config_path}/${env_train_config_file}"
            echo $config_path
            bash ./script/run_dist_train.sh "${env_data_path}/${env_in_data_filename}" "${env_tmp_path}" "${part_path}" "${distributed_num_parts}" "${env_out_path}" "${CONDA_PREFIX}" "${graph_name}" "${graph_CSVDataset_name}" "${config_path}" ${env_ssh_port}
        fi;

        if [ "${distributed_map_save}" = True ]; then
            echo "\nMapping to original graph IDs followed by mapping to CSV file output"
            echo "\nThis may take a while"
            part_path="${env_tmp_path}/partitions"
            config="${env_config_path}/${env_tabular2graph_config_file}"
            echo ${config}
            bash ./script/run_map_save_dist.sh "${env_data_path}/${env_in_data_filename}" "${distributed_num_parts}" "${part_path}" "${env_out_path}" "${config}"
        fi;

    else
        echo -e "\nenv_num_nodes needs to be an integer between 1 and number of machines in the cluster"
    fi;

else
    ## single node docker steps
    if ((${env_num_node} == 1 )); then
        ## Single node docker steps
        docker pull "${env_docker_image}" #This will use the local image if it's up-to-date already and if not it will pull latest
        ERROR_CHECK=$? #$? is a special var to check if previous command returns non-zero exit code/error
        if [ $ERROR_CHECK != 0 ]; then
            echo -e "\nBuilding docker image..."
            cd $repoPath
            #The --pull here is to pull the base image not the target image itself
            docker build -t ${env_docker_image} . \
                --pull \
                --build-arg https_proxy=${https_proxy} \
                --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
                --build-arg HTTP_PROXY=${HTTP_PROXY} \
                --build-arg http_proxy=${http_proxy} \
            #docker build -t ${env_docker_image} --pull -f Dockerfile . #The --pull here is to pull the base image not the target image itself
        else
            echo "PULL successfull"
        fi
        echo -e "\nRun docker image..."
        docker run --shm-size=200g --network host --name gnn\
          -v "${repoPath}":/host \
          -v "${env_data_path}":/DATA_IN \
          -v "${env_out_path}":/DATA_OUT \
          -v "${env_tmp_path}":/GNN_TMP \
          -v "${env_config_path}":/CONFIGS \
          -it ${env_docker_image} ./host/script/run_gnn_wf_docker.sh /CONFIGS/${yamlFileName}
        #docker rm -f gnn
    ## multi node docker steps
    elif ((${env_num_node} > 1 )); then
        echo -e "\nStarting distributed workflow..."
        echo -e "\nCreate ip_config.txt"
        rm ip_config.txt
        for ((i=1; i<=$env_num_node; i++)); do
            ip="env_node_ips_$i"
            echo ${!ip} >> ip_config.txt
        done
        echo -e "\nCreate docker in each worker node"
        for ((i=1; i<=$env_num_node; i++)); do
            ip="env_node_ips_$i"
            echo "\nCreate docker container gnn in ${!ip}"
            ssh ${USER}@${!ip} ${repoPath}/script/start_work_node.sh ${yamlAbsolutePath} ${repoPath}
        done
        
        echo -e "\nRun docker image..."
        docker exec -it gnn /host/script/run_gnn_dist_wf_docker.sh /CONFIGS/${yamlFileName}
        
        for ((i=1; i<=$env_num_node; i++)); do
            ip="env_node_ips_$i"
            ssh ${USER}@${!ip} docker rm -f gnn
        done
    fi;
fi;

end_time=$(date +%s)
total_time=$(( end_time - start_time ))
echo Total time to execute GNN workflow is ${total_time} seconds.
