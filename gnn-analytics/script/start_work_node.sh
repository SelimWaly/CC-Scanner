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
repoPath="$2"

if docker image ls | grep ${env_docker_image} ; then
   echo -e "\n ${env_docker_image} docker image already exists, activating docker environment"
else
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
   fi;
fi;

docker run --privileged=True --shm-size=200g --network host --name gnn\
    -v "${repoPath}":/host \
    -v "${env_data_path}":/DATA_IN \
    -v "${env_out_path}":/DATA_OUT \
    -v "${env_tmp_path}":/GNN_TMP \
    -v "${env_config_path}":/CONFIGS \
    -w / \
    -itd ${env_docker_image}
    