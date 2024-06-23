## Fraud Detection Configuration Files
In this README, you will find a detailed explanation of the important parameters defined in the configuration files for the fraud detection application.

### workflow-config.yaml
| **Param**                | **Description**
| :---                              | :---
| num_node                          | number of machines in the cluster 
| node_ips                          | IP address of cluster machines, the first ip is the master node
| tmp_path                         | a workflow temporary folder will be created under this path; the container path will be `/workspace/tmp`
| data_path               | local data folder to be mounted inside the workflow container; the container path will be `/workspace/data`
| config_path | a local folder which contains the configuration files; the container path will be `/workspace/configs`
| input_data_path  | relative path of the input data, it should be a subfolder of data_path
| input_data_format | the input data format, e.g. csv
| output_data_path | relative path of the output data, it should be a subfolder of data_path
| output_data_format | the output data format, e.g. csv
| dp_config_file | the name of the data preprocessing yaml file, it should be found under config_path
| dp_framework | data preprocessing backend, either pandas or spark
| train_config_file | the name of the model training yaml file, it should be found under config_path
| train_framework | model training backend, either pandas or spark
| test_backend | either xgboost-native or xgboost-onedal, using xgboost-onedal for performance acceleration
| ray_params | only needed if you do multi-node xgboost training, based on your cluster env, choose appropriate numbers of num_actors and cpus_per_actor
 
### data-preprocessing.yaml
| **Param**                | **Description**
| :---                              | :---
| normalize_feature_names           | normalize feature names, e.g. replace existing characters or lowercase the names 
| categorify                        | the same as the `astype('category')` operation, the source column should be on the left  
| strip_chars                       | strip chars for a given column  
| combine_cols               | make new column based on existing columns, e.g. for 2 string columns, concatenate the values
| time_to_seconds | extract seconds from time values
| change_datatype  | change the data type of existing columns
| min_max_normalization | normalize column values based on min and max values
| one_hot_encoding | True means to drop the column after one-hot encoding, use False for not dropping
| multi_hot_encoding | True means to drop the column after multi-hot encoding, use False for not dropping
| add_constant_feature | add a new column with constant values
| string_to_list | split the values in a string column to a list 
| modify_on_conditions | modify the values in a column based on the given conditions, pls use `df` for dataframe 
| define_variable | define new variables based on the given conditions
| custom_rules | create 2 datasets called train and test using the given conditions
| target_encoding | do target encoding for the given feature columns and target column 


### model-training.yaml
| **Param**                | **Description**
| :---                              | :---
| target_col           | target column for model training
| ignore_cols                        | ignore these columns from the raw data 
| data_split                       | split data based on the given conditions 
| model_type               | currently, only xgboost is supported 
| model_params | the xgboost model parameters
| training_params  | the xgboost training parameters, e.g. num_boost_round
| test_metric | test_metric should be the same with eval_metric in model_params
| search_mode | either min or max
| num_trials | number of training trials for HPO