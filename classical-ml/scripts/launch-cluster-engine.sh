#!/bin/bash

yaml="$1"
repo_name="$2"
cluster_engine="$3"
is_master="$4"
mode="$5"
master_port="$6"

script_path=$(dirname "$(realpath $0)")
wf_abs_path=$(dirname $script_path)
lib_abs_path=$(dirname $wf_abs_path)

yaml_name=$(basename $yaml)
    
if [ "$mode" == 'docker' ]; then
    yaml_path=/workspace/configs/$yaml_name
else 
    yaml_path=$yaml 
fi 

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

if [ "$mode" == 'docker' ]; then
    config_path=/workspace/$repo_name/configs
    tmp_path=/workspace/tmp
else 
    config_path=$wf_abs_path/configs
    tmp_path=$env_tmp_path/wf-tmp
    HADOOP_HOME=$lib_abs_path/lib/hadoop-3.3.3
    JAVA_HOME=$lib_abs_path/lib/jdk1.8.0_201
    SPARK_HOME=$lib_abs_path/lib/spark-3.3.1-bin-hadoop3s
fi 

if [ "$cluster_engine" == 'spark' ]; then
    
    spark_version=`echo $SPARK_HOME | cut -d "-" -f 2`
    
    if [ "$mode" == 'docker' ]; then

        cp $config_path/hadoop-env.sh $HADOOP_HOME/etc/hadoop/hadoop-env.sh && \
        cp $config_path/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml && \
        cp $config_path/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml && \
        cp $config_path/mapred-site.xml $HADOOP_HOME/etc/hadoop/mapred-site.xml && \
        cp $config_path/yarn-site.xml $HADOOP_HOME/etc/hadoop/yarn-site.xml && \
        cp $config_path/slaves $HADOOP_HOME/etc/hadoop/slaves && \
        cp $config_path/workers $HADOOP_HOME/etc/hadoop/workers && \
        cp $config_path/spark-env.sh $SPARK_HOME/conf/spark-env.sh && \
        cp $config_path/spark-defaults.conf $SPARK_HOME/conf/spark-defaults.conf      
        
        sed -i 's@hadoop-leader@'"$env_node_ips_1"'@'  $HADOOP_HOME/etc/hadoop/core-site.xml && \
        sed -i 's@hadoop-leader@'"$env_node_ips_1"'@'  $HADOOP_HOME/etc/hadoop/hdfs-site.xml && \
        sed -i 's@hadoop-leader@'"$env_node_ips_1"'@'  $HADOOP_HOME/etc/hadoop/yarn-site.xml && \
        sed -i 's@hadoop-leader@'"$env_node_ips_1"'@'  $HADOOP_HOME/etc/hadoop/mapred-site.xml && \
        sed -i 's@hadoop-leader@'"$env_node_ips_1"'@'  $SPARK_HOME/conf/spark-defaults.conf && \
        sed -i 's@hadoop-leader@'"$env_node_ips_1"'@'  $SPARK_HOME/conf/spark-env.sh && \
        sed -i 's@hadoop-leader@'"$env_node_ips_1"'@'  $HADOOP_HOME/etc/hadoop/slaves && \
        sed -i 's@hadoop-leader@'"$env_node_ips_1"'@'  $HADOOP_HOME/etc/hadoop/workers  
    fi 

    if [ -z "${env_node_ips_2}" ]; then 

        echo -e "\nformat namenode..."
        (sleep 10; echo y) | $HADOOP_HOME/bin/hdfs namenode -format
        echo -e "\nsetting up single-node Spark cluster..."   
        echo -e "\nstart HDFS..."
        $HADOOP_HOME/sbin/start-dfs.sh
        echo -e "\n"

        echo -e "\nstart spark master..."
        source $SPARK_HOME/conf/spark-env.sh 
        $SPARK_HOME/sbin/start-master.sh
            
        echo -e "\nstart spark worker..."
        $SPARK_HOME/sbin/start-worker.sh spark://$env_node_ips_1:7077 

        echo -e "\ncreate spark history folder..."
        hdfs dfs -mkdir -p /spark/history

        echo -e "\nstart spark history..."
        $SPARK_HOME/sbin/start-history-server.sh

    else

        for ((i=2; i<=$env_num_node; i++)); do
            worker_ip="env_node_ips_$i"
            echo "" >> $HADOOP_HOME/etc/hadoop/slaves 
            echo "${!worker_ip}" >> $HADOOP_HOME/etc/hadoop/slaves 
            echo "" >> $HADOOP_HOME/etc/hadoop/workers 
            echo "${!worker_ip}" >> $HADOOP_HOME/etc/hadoop/workers 
        done

        if [ $is_master = "1" ]
        then 
            echo -e "\nformat namenode..."
            (sleep 10; echo y) | $HADOOP_HOME/bin/hdfs namenode -format
        else 
            echo -e "\nformat datanode..."
            (sleep 10; echo y) | $HADOOP_HOME/bin/hdfs datanode -format
        fi

        echo "starting hadoop..."
        $HADOOP_HOME/sbin/start-dfs.sh
            
        echo -e "\nstart spark master..."
        source $SPARK_HOME/conf/spark-env.sh 

        if [ $is_master = "1" ]; then
        
            $SPARK_HOME/sbin/start-master.sh
            echo -e "\ncreate spark history folder..."
            hdfs dfs -mkdir -p /spark/history

        fi 

        if [[ $SPARK_VERSION == '3.3.0' ]]
        then 
            echo -e "\nstart spark worker..."
            $SPARK_HOME/sbin/start-worker.sh spark://$env_node_ips_1:7077
        else 
            echo -e "\nstart spark worker..."
            $SPARK_HOME/sbin/start-slave.sh spark://$env_node_ips_1:7077
        fi 

    fi
    
    if [ $is_master = "0" ]; then
        echo -e "\ntest with the Pi Example...\n"
        $SPARK_HOME/bin/spark-submit --class org.apache.spark.examples.SparkPi --master spark://$env_node_ips_1:7077 $SPARK_HOME/examples/jars/spark-examples_2.12-$spark_version.jar 10
    fi 

else 

    if [ $is_master = "1" ]; then
        ray start --head --port=6379 --temp-dir=$tmp_path/ray
    else 
        ray start --address=$master_port --temp-dir=$tmp_path/ray
    fi 

fi 