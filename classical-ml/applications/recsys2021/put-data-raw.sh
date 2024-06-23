echo -e "\ncreate recsys2021 HDFS data folder"

hdfs dfs -mkdir -p /recsys2021/oridata
hdfs dfs -mkdir -p /recsys2021/datapre_stage1
hdfs dfs -mkdir -p /recsys2021/datapre_stage2
hdfs dfs -mkdir -p /recsys2021/models
hdfs dfs -mkdir -p /recsys2021/results

echo -e "\nload data into HDFS folder"
hdfs dfs -put /workspace/data/train /recsys2021/oridata
hdfs dfs -put /workspace/data/valid /recsys2021/oridata

echo -e "\ncheck whether data is there"
hdfs dfs -ls /recsys2021/oridata