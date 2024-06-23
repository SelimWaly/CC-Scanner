#!/bin/bash

yaml_path="$(realpath $1)"

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

eval $(parse_yaml $yaml_path)

wf_abs_path=$(dirname "$(realpath $0)")
host_name=$(hostname -I | awk '{print $1}')
curr_time=$(date +%Y-%m-%d-%H-%M)
wf_tmp_path="$env_tmp_path/wf-session-${curr_time}"
repo_name=$(basename $wf_abs_path)

if [ "$data_preprocess_dp_framework" == "spark" ] && [ "$training_train_framework" == "spark" ]; then 
    cluster_engine='spark'
elif [ "$data_preprocess_dp_framework" == 'spark' ] && [ -z "$training_train_framework" ]; then 
    cluster_engine='spark'
elif [ -z "$data_preprocess_dp_framework" ] && [ "$training_train_framework" == 'spark' ]; then 
    cluster_engine='spark'
else
    cluster_engine='ray'
fi 


if [ -z "${env_node_ips_2}" ]; then   
    echo -e "\nsetting up single-node environment..."
    
    if [ "$env_node_ips_1" == "$host_name" ]; then
        bash $wf_abs_path/scripts/create-wf-tmp-folders.sh "$wf_tmp_path"
        bash $wf_abs_path/scripts/launch-wf-containers.sh "$env_data_path" "$wf_tmp_path" "$env_config_path" "$wf_abs_path" hadoop-leader
        docker exec hadoop-leader mkdir -p $(dirname $yaml_path)
        docker cp $yaml_path hadoop-leader:$yaml_path
        if [ "$cluster_engine" == 'ray' ]; then
            echo "start workflow engine..."
            docker exec hadoop-leader python start-workflow.py --config-file ${yaml_path} --mode 1
            echo "remove workflow containers..."
            docker rm -f hadoop-leader
        elif [ "$cluster_engine" == 'spark' ]; then 
            docker exec hadoop-leader pip install torch 
            docker exec hadoop-leader bash scripts/launch-cluster-engine.sh "$yaml_path" "$repo_name" "$cluster_engine" 1 docker
        else
            echo "Unknown Cluster Engine"
        fi      
    
    elif [ "$env_node_ips_1" == "localhost" ]; then
        env_node_ips_1=$host_name
        bash $wf_abs_path/scripts/create-wf-tmp-folders.sh "$wf_tmp_path"
        bash $wf_abs_path/scripts/launch-wf-containers.sh "$env_data_path" "$wf_tmp_path" "$env_config_path" "$wf_abs_path" hadoop-leader
        docker exec hadoop-leader mkdir -p $(dirname $yaml_path)
        docker cp $yaml_path hadoop-leader:$yaml_path
        if [ "$cluster_engine" == 'ray' ]; then
            echo "start workflow engine..."
            docker exec hadoop-leader python start-workflow.py --config-file ${yaml_path} --mode 1
            echo "remove workflow containers..."
            docker rm -f hadoop-leader
        elif [ "$cluster_engine" == 'spark' ]; then 
            docker exec hadoop-leader pip install torch 
            docker exec hadoop-leader bash scripts/launch-cluster-engine.sh "$yaml_path" "$repo_name" "$cluster_engine" 1 docker
        else
            echo "Unknown Cluster Engine"
        fi      
        
    else
        tar_path="$(dirname "$wf_abs_path")"
        curr_time4=$(date +%Y-%m-%d-%H-%M)
        scp -q -r ${wf_abs_path} ${env_node_ips_1}:${tar_path}/ &> ${wf_tmp_path}/logs/scp-${curr_time4}.log
        scp -q -r ${env_config_path} ${env_node_ips_1}:${env_config_path}/ &> ${wf_tmp_path}/logs/scp-${curr_time4}.log
        scp -q -r $yaml_path ${env_node_ips_1}:$yaml_path &> ${wf_tmp_path}/logs/scp-${curr_time4}.log

        echo -e "\nssh to master ${master_ip}"
        ssh ${env_node_ips_1} "
            cd ${wf_abs_path}
            bash ./scripts/create-wf-tmp-folders.sh ${wf_tmp_path}
            bash ./scripts/launch-wf-containers.sh ${env_data_path} ${wf_tmp_path} ${env_config_path} ${wf_abs_path} hadoop-leader
            docker exec hadoop-leader mkdir -p $(dirname $yaml_path)
            docker cp $yaml_path hadoop-leader:$yaml_path
        "

        if [ "$cluster_engine" == 'ray' ]; then
            ssh ${env_node_ips_1} " 
                docker exec hadoop-leader python start-workflow.py --config-file ${yaml_path} --mode 1
                echo "shut down workflow containers..."
                docker rm -f hadoop-leader
            "
        elif [ "$cluster_engine" == 'spark' ]; then 
            ssh ${env_node_ips_1} "
                docker exec hadoop-leader pip install torch 
                docker exec hadoop-leader bash scripts/launch-cluster-engine.sh ${yaml_path} ${repo_name} ${cluster_engine} 0 docker
            "
        else
            echo "Unknown Cluster Engine"
        fi      
    fi

else 
    echo -e "\nsetting up multi-node environment..."

    echo -e "\non master node..."
    if [ "$host_name" == "$env_node_ips_1" ]; then
        bash $wf_abs_path/scripts/create-wf-tmp-folders.sh "$wf_tmp_path"
        bash $wf_abs_path/scripts/launch-wf-containers.sh "$env_data_path" "$wf_tmp_path" "$env_config_path" "$wf_abs_path" hadoop-leader
        docker exec hadoop-leader mkdir -p $(dirname $yaml_path)
        docker cp $yaml_path hadoop-leader:$yaml_path
    else
        tar_path="$(dirname "$wf_abs_path")"
        curr_time2=$(date +%Y-%m-%d-%H-%M)
        scp -q -r ${wf_abs_path} ${env_node_ips_1}:${tar_path}/ &> tee -a ${wf_tmp_path}/logs/scp-${curr_time2}.log
        scp -q -r ${env_config_path} ${env_node_ips_1}:${env_config_path}/ &> tee -a ${wf_tmp_path}/logs/scp-${curr_time2}.log
        scp -q -r $yaml_path ${env_node_ips_1}:$yaml_path &> tee -a ${wf_tmp_path}/logs/scp-${curr_time2}.log
        container_name="hadoop-leader"
        echo -e "\nssh to master ${master_ip}"
        ssh ${env_node_ips_1} "
            cd ${wf_abs_path}
            bash ./scripts/create-wf-tmp-folders.sh ${wf_tmp_path}
            bash ./scripts/launch-wf-containers.sh ${env_data_path} ${wf_tmp_path} ${env_config_path} ${wf_abs_path} ${container_name}
            docker exec ${container_name} mkdir -p $(dirname $yaml_path)
            docker cp $yaml_path ${container_name}:$yaml_path
        "
    fi 
    
    echo -e "\non worker node..."
    for ((i=2; i<=$env_num_node; i++)); do
        worker_ip="env_node_ips_$i"
        worker_num=$((i-1))
        container_name="hadoop-worker$worker_num"
        tar_path="$(dirname "$wf_abs_path")"
        curr_time3=$(date +%Y-%m-%d-%H-%M)
        scp -q -r ${wf_abs_path} ${!worker_ip}:${tar_path}/ &> ${wf_tmp_path}/logs/scp-${curr_time3}.log
        scp -q -r ${env_config_path} ${!worker_ip}:${env_config_path}/ &> ${wf_tmp_path}/logs/scp-${curr_time3}.log
        scp -q -r $yaml_path ${!worker_ip}:$yaml_path &> ${wf_tmp_path}/logs/scp-${curr_time3}.log
        ssh ${!worker_ip} "
            cd ${wf_abs_path}
            bash ./scripts/create-wf-tmp-folders.sh ${wf_tmp_path}
            bash ./scripts/launch-wf-containers.sh ${env_data_path} ${wf_tmp_path} ${env_config_path} ${wf_abs_path} ${container_name}
            docker exec ${container_name} mkdir -p $(dirname $yaml_path)
            docker cp $yaml_path ${container_name}:$yaml_path
        "
    done


    if [ "$cluster_engine" == 'ray' ]; then
        echo -e "\nlaunch ray cluster inside the containers..."
        if [ "$host_name" == "$env_node_ips_1" ]; then
            docker exec hadoop-leader bash scripts/launch-cluster-engine.sh ${yaml_path} ${repo_name} ${cluster_engine} 1 docker 
        else 
            ssh ${env_node_ips_1} "
                docker exec hadoop-leader bash scripts/launch-cluster-engine.sh ${yaml_path} ${repo_name} ${cluster_engine} 1 docker
            "
        fi 

        for ((i=2; i<=$env_num_node; i++)); do
            worker_ip="env_node_ips_$i"
            worker_num=$((i-1))
            worker_name="hadoop-worker$worker_num"
            master_port="$env_node_ips_1:6379"
            ssh ${!worker_ip} "
                docker exec $worker_name bash scripts/launch-cluster-engine.sh ${yaml_path} ${repo_name} ${cluster_engine} 0 docker ${master_port}
            "
        done

        echo -e "\nstart workflow on master...\n"
        if [ "$host_name" == "$env_node_ips_1" ]; then
            docker exec hadoop-leader python start-workflow.py --config-file ${yaml_path} --mode 1
            if [ ! -z "${data_preprocess_input_data_path}" ]; then
                echo "collecting files from nodes to master..."
                docker exec hadoop-leader python src/utils/collect_modin_files.py ${yaml_path} 1
            fi
        else 
            ssh ${env_node_ips_1} "
                cd ${wf_abs_path}
                docker exec -i hadoop-leader python start-workflow.py --config-file ${yaml_path} --mode 1
            "
        fi 

        echo -e "\nremove workflow containers...\n"

        if [ "$host_name" == "$env_node_ips_1" ]; then
            docker rm -f hadoop-leader 
        else 
            ssh ${env_node_ips_1} "
                docker rm -f hadoop-leader
                rm -fr ${wf_abs_path}
            " 
        fi 

        for ((i=2; i<=$env_num_node; i++)); do
            worker_ip="env_node_ips_$i"
            worker_num=$((i-1))
            worker_name="hadoop-worker$worker_num"
            ssh ${!worker_ip} "
                docker rm -f ${worker_name}
                rm -fr ${wf_abs_path}
            "
        done

    elif [ "$cluster_engine" == 'spark' ]; then 

        echo -e "\nlaunch spark cluster inside the containers..."
        if [ "$host_name" == "$env_node_ips_1" ]; then
            docker exec hadoop-leader pip install torch 
            docker exec hadoop-leader bash scripts/launch-cluster-engine.sh ${yaml_path} ${repo_name} ${cluster_engine} 1 docker
        else 
            ssh ${env_node_ips_1} "
                docker exec hadoop-leader pip install torch 
                docker exec hadoop-leader bash scripts/launch-cluster-engine.sh ${yaml_path} ${repo_name} ${cluster_engine} 1 docker
            "
        fi 

        for ((i=2; i<=$env_num_node; i++)); do
            worker_ip="env_node_ips_$i"
            worker_num=$((i-1))
            worker_name="hadoop-worker$worker_num"
            master_port="$env_node_ips_1:6379"
            ssh ${!worker_ip} "
                docker exec $worker_name pip install torch 
                docker exec $worker_name bash scripts/launch-cluster-engine.sh ${yaml_path} ${repo_name} ${cluster_engine} 0 docker ${master_port}
            "
        done
    else 
        echo "Unkown Cluster Engine"
    fi   
fi
