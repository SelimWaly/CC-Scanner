## RecSys2021 Workflow Application
Different than the fraud detection use case which uses the full config-driven approach, the recsys2021 uses the script-driven approach. This means that users only need to write a minimal `workflow-config.yaml` and execute python scripts where the existing workflow APIs can be utilized. This tutorial will show you how to run the RecSys2021 application using the scripts under the recsys2021 application folders.  

### Workflow-config.yaml
For the RecSys2021 application, only the `workflow-config.yaml` is needed. As can be seen in the [example](application/recsys2021/workflow-config.yaml), users only need to specify the `env` and then leave all params blank in the `data_preprocess` session expect the `dp_framework`.

### Run Data Generator
```bash 
python applications/recsys2021/generate_recsys_data.py /workspace/data
```
This will create 2 folders with the name `train` and `valid` under `/workspcae/data`.

### Run Data Preprocessing
```bash 
# enter the workflow container
docker exec -it hadoop-leader bash
# load data into HDFS
./applications/put-data-raw.sh
# for data preprocessing
python applications/recsys2021/recsys_datapre.py train [master_ip] [is_local] # use 1 for single-node and 0 for multi-node
python applications/recsys2021/recsys_datapre.py train valid_stage1 [is_local]
python applications/recsys2021/recsys_datapre.py train valid_stage2 [is_local]
```

### Run Model Training
```bash 
# enter the workflow container
docker exec -it hadoop-leader bash 
# for training
python applications/recsys2021/recsys_model_training.py --config-file applications/recsys2021/workflow-config.yaml 
```
