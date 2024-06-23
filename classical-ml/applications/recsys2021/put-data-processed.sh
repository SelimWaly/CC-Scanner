echo -e "\ncreate recsys2021 HDFS data folder"
  
hdfs dfs -mkdir -p /recsys2021/datapre_stage1
hdfs dfs -mkdir -p /recsys2021/datapre_stage2
hdfs dfs -mkdir -p /recsys2021/models

echo -e "\nload data into HDFS folder"
hdfs dfs -put /workspace/data/processed_data/stage1_train /recsys2021/datapre_stage1
hdfs dfs -put /workspace/data/processed_data/stage1_valid /recsys2021/datapre_stage1
hdfs dfs -put /workspace/data/processed_data/stage2_train /recsys2021/datapre_stage2
hdfs dfs -put /workspace/data/processed_data/stage2_valid /recsys2021/datapre_stage2

echo -e "\ncheck whether data is there"
hdfs dfs -ls /recsys2021/datapre_stage1
hdfs dfs -ls /recsys2021/datapre_stage2