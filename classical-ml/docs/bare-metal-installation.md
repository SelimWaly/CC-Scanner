## Bare Metal Installation Guide  
The following guide assumes that you are under the `/home` directory on your linux machine and the workflow will create a temporary folder named `wf-tmp` under `/home`. Please note that `/home` corresponds to value of `tmp_path` in your `workflow-config.yaml` file. See an example of `workflow-config.yaml`, please check out the `applications/fraud_detection` folder.

### 1. Set Up System Software

Our examples use the conda package and environment on your local computer. If you don't already have conda installed, see the [Conda Linux installation instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).


### 2. Prepare Workflow Environment
Pls make sure that you are the `root` user of your cluster machines, because running Hadoop/Spark cluster requires the root privilege.

#### 2.1 Set Up Hadoop/Spark Environment

1. create folders on master node
```bash
# create project folder
mkdir dcml
cd dcml
# create folder to save Spark, Hadoop and Java source code 
mkdir lib
cd lib
```

2. download the binary files of Spark, Hadoop, Java 
```bash
# download
wget --no-check-certificate https://dlcdn.apache.org/hadoop/common/hadoop-3.3.3/hadoop-3.3.3.tar.gz
wget --no-check-certificate https://archive.apache.org/dist/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz
wget --no-check-certificate https://repo.huaweicloud.com/java/jdk/8u201-b09/jdk-8u201-linux-x64.tar.gz

# unpack 
tar -zxvf hadoop-3.3.3.tar.gz
tar -zxvf spark-3.3.1-bin-hadoop3.tgz
tar -zxvf jdk-8u201-linux-x64.tar.gz

# remove the tar files
rm hadoop-3.3.3.tar.gz
rm spark-3.3.1-bin-hadoop3.tgz
rm jdk-8u201-linux-x64.tar.gz
```

3. configure Hadoop config file
```bash
cd hadoop-3.3.3

# edit the hadoop-env.sh file
vim etc/hadoop/hadoop-env.sh

# add the following to the end of this file and save
export JAVA_HOME=/home/dcml/lib/jdk1.8.0_201
export HDFS_NAMENODE_USER=root
export HDFS_DATANODE_USER=root
export HDFS_SECONDARYNAMENODE_USER=root
export YARN_RESOURCEMANAGER_USER=root
export YARN_NODEMANAGER_USER=root

# edit the file
vim etc/hadoop/workers

# add your the hostnames of your cluster machines here, e.g. if you have `mlp-opa-clx-4021` as your master node and `mlp-opa-clx-4025` as your worker node
mlp-opa-clx-4021
mlp-opa-clx-4025

# edit the core-site.xml file
vim etc/hadoop/core-site.xml

# add the following to the configuration brackets and save
    <property>
            <name>hadoop.tmp.dir</name>
            <value>/home/wf-tmp/hdfs/tmp</value> # be careful about value in this line 
        </property>
        <property>
            <name>fs.defaultFS</name> 
            <value>hdfs://mlp-opa-clx-4021:9000</value> # be careful about value in this line 
    </property>

# edit the hdfs-site.xml file
vim etc/hadoop/hdfs-site.xml

#  add the following to the configuration brackets and save 
         <property>
            <name>dfs.replication</name>
            <value>1</value>
        </property>
     
        <property>
        <name>dfs.namenode.name.dir</name>
        <value>/home/wf-tmp/hdfs/nn</value> # be careful about value in this line 
      </property>
     
       <property>
        <name>dfs.datanode.data.dir</name>
        <value>/home/wf-tmp/hdfs/dn</value> # be careful about value in this line 
      </property>
     
        <property>
            <name>dfs.namenode.http-address</name>
            <value>mlp-opa-clx-4021:9870</value> # be careful about value in this line 
        </property>
        <property>
            <name>dfs.namenode.secondary.http-address</name>
            <value>mlp-opa-clx-4021:9868</value> # be careful about value in this line 
    </property>

# edit the mapred-site.xml file
vim etc/hadoop/mapred-site.xml

#  add the following to the configuration brackets and save 
  <property>
            <name>mapreduce.framework.name</name>
            <value>yarn</value>
       </property>
       <property>
            <name>mapreduce.jobhistory.address</name>
            <value>mlp-opa-clx-4021:10020</value> # be careful about value in this line
       </property>
       <property>
            <name>mapreduce.jobhistory.webapp.address</name>
            <value>mlp-opa-clx-4021:19888</value> # be careful about value in this line
       </property>
 ```

4. set the Hadoop&Spark env variables in ./bashrc
```bash
vim ~/.bashrc
 
# append the following and save
HADOOP_HOME=/home/dcml/lib/hadoop-3.3.3
JAVA_HOME=/home/dcml/lib/jdk1.8.0_201
JRE_HOME=${JAVA_HOME}/jre
SPARK_HOME=/home/dcml/lib/spark-3.3.1-bin-hadoop3
PYSPARK_PYTHON=<> #pls use the command `which python` to find out the python path on your machine
PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin
PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip:$PYTHONPATH
HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
HADOOP_OPTS="$HADOOP_OPTS -Djava.library.path=$HADOOP_HOME/lib/native"
LD_LIBRARY_PATH=$HADOOP_HOME/lib/native
 
source ~/.bashrc

# check whether it works
java -version
hadoop version
```

4. enable password-less ssh 

You need to ensure password-less ssh among your cluster machines. For detailed instruction, pls refer to this [post](http://www.drugdesign.gr/blog/how-to-setup-passwordless-ssh-access-between-nodes-in-a-cluster).


5. clone workflow repo 
```bash
cd /home/dcml
git clone https://github.com/intel/recommender-system-with-distributed-classical-ml.git
```

6. copy files over to worker node 
```bash
scp -r /home/dcml <worker_ip>:/home
scp ~/.bashrc <worker_ip>:~
```
on worker nodes, run 
```bash
source ~/.bashrc
```

7. create conda python environment and prepare temporary folder
on each machine, run the following
```bash
cd /home/dcml/recommender-system-with-distributed-classical-ml
./scripts/create-wf-tmp-folders.sh /home/wf-tmp
./scripts/create-conda-env.sh 
conda activate dcml 
```

### 3. Run Workflow 

### 3.1 Launch Cluster 
If you are using Spark as cluster engine:
```bash 
# on master
./scripts/launch-cluster-engine.sh <yaml_path> recommender-system-with-distributed-classical-ml spark 1 bare-metal

#on worker node  
./scripts/launch-cluster-engine.sh <yaml_path> recommender-system-with-distributed-classical-ml spark 0 bare-metal
```

If you are using ray as cluster engine:
```bash 
#on master 
ray start --head --port=6379 --temp-dir=/home/wf-tmp/ray
#on worker
ray start --head --address=<master_ip:6379> --temp-dir=/home/wf-tmp/ray
```

### 3.2 Start Workflow
If you are using the config-driven execution mode of the workflow, run
```bash
python start-workflow.py --config-file <workflow_config_path> --mode 0
```

## Expected Output
If the workflow executes successfully, you should see the message `shut down cluster...` and the cluster node names, e.g. `hadoop-master` and `hadoop-slave1`  at the end of the command line output. 
