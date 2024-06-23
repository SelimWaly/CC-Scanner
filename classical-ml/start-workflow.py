"""
Copyright [2022-23] [Intel Corporation]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#!/usr/bin/env python
# coding: utf-8
import argparse
import os, time, gc, sys, glob
import pandas as pd 
import numpy as np
import yaml 
import time 
from src.utils.data_utils import *

very_start = time.time()

PATH_HOME = os.path.dirname(os.path.realpath(__file__))

class WFProcessor:

    def __init__(self, file_name, mode): 
        
        if mode == 1:
            self.log_path = '/workspace/tmp/logs'
            self.data_path = '/workspace/data'
            self.tmp_path = '/workspace/tmp'
            self.model_save_path = os.path.join(self.tmp_path, 'models')
            self.config_path = '/workspace/configs'
            with open(os.path.join(self.config_path,os.path.basename(file_name)),'r') as file:
                config = yaml.safe_load(file)
        else:
            with open(file_name,'r') as file:
                config = yaml.safe_load(file)
            self.data_path = config['env']['data_path']
            self.tmp_path = os.path.join(config['env']['tmp_path'], 'wf-tmp')
            self.log_path = os.path.join(self.tmp_path, 'logs')
            self.model_save_path = os.path.join(self.tmp_path, 'models')
            self.config_path = config['env']['config_path']
        
        self.num_node = config['env']['num_node'] 
        self.is_multi_nodes = True if self.num_node > 1 else False  
        self.worker_ips = config['env']['node_ips'][1:]
        
        try:
            self.raw_data_path = os.path.join(self.data_path, config['data_preprocess']['input_data_path']) 
            self.raw_data_format = config['data_preprocess']['input_data_format'] 
            dp_config_file = os.path.join(self.config_path, config['data_preprocess']['dp_config_file']) 
            self.dp_framework = config['data_preprocess']['dp_framework']
            self.processed_data_path = os.path.join(self.data_path, config['data_preprocess']['output_data_path']) 
            self.processed_data_format = config['data_preprocess']['output_data_format']
            self.read_data_processing_steps(dp_config_file)
            self.identify_dp_engine()
            self.has_dp = True
        except Exception as e: 
            print('Failed to read data preprocessing steps. This is either due to wrong parameters defined in the config file as shown: '+ str(e) 
                    + ' or there is no need for data preprocessing.')
            self.has_dp = False
        try:
            self.train_data_path = os.path.join(self.data_path, config['training']['input_data_path']) 
            self.train_data_format = config['training']['input_data_format']
            train_config_file = os.path.join(self.config_path, config['training']['train_config_file']) 
            self.train_framework = config['training']['train_framework']
            self.test_backend = config['training']['test_backend']
            self.read_training_configs(train_config_file)
            try:
                self.ray_params = config['training']['ray_params']
            except:
                self.ray_params = None 
            self.has_training = True
        except Exception as e: 
            print('Failed to read model training configurations. This is either due to wrong parameters defined in the config file as shown: '+ str(e) 
                    + ' or there is no need for model training.')
            self.has_training = False
        try: 
            self.raw_data_path = os.path.join(self.data_path, config['end2end_training']['input_data_path']) 
            self.raw_data_format = config['end2end_training']['input_data_format']
            dp_config_file = os.path.join(self.config_path, config['end2end_training']['dp_config_file'])
            self.dp_framework = config['end2end_training']['framework']
            self.train_framework = config['end2end_training']['framework']
            self.read_data_processing_steps(dp_config_file)
            self.identify_dp_engine()
            self.train_data_path=None
            self.train_data_format=None 
            train_config_file = os.path.join(self.config_path, config['end2end_training']['train_config_file'])
            self.read_training_configs(train_config_file)
            self.test_backend = config['end2end_training']['test_backend']
            try:
                self.ray_params = config['end2end_training']['ray_params']
            except:
                self.ray_params = None 
            self.in_memory = True
        except Exception as e: 
            print("Failed to read end2end training configurations. This is either due to wrong parameters defined in the config file as shown: "+ str(e)
                  + " or there is no need for End-to-End training.")
            self.in_memory = False
        self.cluster_engine = None
    
    def identify_dp_engine(self):
        if self.dp_framework == 'pandas' and self.is_multi_nodes == False:
            self.dp_engine = 'pandas'
        elif self.dp_framework == 'pandas' and self.is_multi_nodes == True:
            self.dp_engine = 'modin'
        else:
            self.df_engine = 'spark'
        
    def prepare_env(self):
        if self.is_multi_nodes: 
            print("enter distributed mode...")
            if self.has_dp and self.has_training:
                if self.dp_framework == 'spark' and self.train_framework == 'spark':
                    self.cluster_engine='spark'
                else:
                    self.cluster_engine='ray'
            elif self.has_dp and not self.has_training: 
                if self.dp_framework == 'spark':
                    self.cluster_engine='spark'
                else:
                    self.cluster_engine='ray'
            elif not self.has_dp and self.has_training: 
                if self.train_framework == 'spark':
                    self.cluster_engine='spark'
                else:
                    self.cluster_engine='ray'
            elif self.in_memory:
                if self.dp_framework == 'spark':
                    self.cluster_engine='spark'
                else:
                    self.cluster_engine='ray'
            else:
                print("Check your workflow config file. Program End.")
                sys.exit()
            if self.cluster_engine == 'ray':
                print("initializing ray cluster...")
                import ray 
                ray.init('auto', runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}}, log_to_driver=False)
            if self.cluster_engine == 'spark':
                raise NotImplementedError("spark cluster engine is currently not supported in config-style usage of the workflow.")
        else:
            print("enter single-node mode...")
            if not self.has_dp and not self.has_training and not self.in_memory:
                print("Program End.")
                sys.exit()
            
    def read_data_processing_steps(self, dp_config_file):
        with open(dp_config_file, 'r') as file:
            dp_steps = yaml.safe_load(file)
        
        self.pre_splitting_steps = dp_steps['pre_splitting_transformation']
        self.data_splitting_rule = dp_steps['data_splitting']
        self.post_splitting_steps = dp_steps['post_splitting_transformation']
    
    def read_training_configs(self, train_config_file):
        with open(train_config_file, 'r') as file:
            train_configs = yaml.safe_load(file)
        
        self.train_data_spec = train_configs['data_spec']
        try:
            self.hpo_spec = train_configs['hpo_spec']
            self.hpo_needed = True
        except:
            self.hpo_spec = None 
            self.hpo_needed = False
            print("no need for HPO")
        try:
            self.train_model_spec = train_configs['model_spec']
            self.hpo_needed = False
        except:
            self.train_model_spec = None 
            self.hpo_needed = True
            print("no need for training")
        
        if self.hpo_spec is None and self.train_model_spec is None:
            print("none of the hpo_spec and model_spec is specified. Program End.")
            sys.exit()
        elif self.hpo_spec is not None and self.train_model_spec is not None:
            print("Pls specify either hpo_spec or model_spec. Both are not accepted. Program End.")
            sys.exit()

    def read_raw_data(self):
        print('reading raw data...')
        if self.raw_data_format == 'csv':
            self.data = read_csv_files(self.raw_data_path, engine=self.dp_engine)

    def read_train_data(self):
        print('reading training data...')
        if self.train_data_format == 'csv':
            self.data = read_csv_files(self.train_data_path, engine='pandas')

    def pre_splitting_transform(self):
        print("transform pre-splitting data...")
        if self.dp_engine != 'spark':
            from src.preprocessing.pandas.pre_splitting_transformation import PreSplittingTransformer
            pre_splitting_transformer = PreSplittingTransformer(self.data, self.pre_splitting_steps, self.dp_engine)
            self.data = pre_splitting_transformer.process()
        else:
            raise NotImplementedError("currently only pandas-based data preprocessing is supported")

    def split_data(self):
        print('splitting data...')
        if self.dp_engine != 'spark':
            if self.dp_engine == 'modin':
                import modin.pandas as pd 
            elif self.dp_engine == 'pandas':
                import pandas as pd 
            from src.preprocessing.pandas.data_splitting import DataSplitter 
            data_splitter = DataSplitter(self.data, self.data_splitting_rule)
            self.train_data, self.test_data = data_splitter.process()
            self.data = None 
        else:
            raise NotImplementedError("currently only pandas-based data preprocessing is supported")
        
    def post_splitting_transform(self):
        print("transform pre-splitting data...")
        if self.dp_engine != 'spark':
            from src.preprocessing.pandas.post_splitting_transformation import PostSplittingTransformer
            pre_splitting_transformer = PostSplittingTransformer(self.train_data, self.test_data, self.post_splitting_steps, self.dp_engine)
            self.train_data, self.test_data = pre_splitting_transformer.process()
        else:
            raise NotImplementedError("currently only pandas-based data preprocessing is supported")
        
    def save_processed_data(self):
        print('saving data...')
        if self.dp_engine != 'spark':
            if self.dp_engine == 'modin':
                import modin.pandas as pd 
            else:
                import pandas as pd 
            if self.processed_data_format == 'csv':
                data = pd.concat([self.train_data, self.test_data])
                data.to_csv(self.processed_data_path+'/processed_data.csv', index=False)
                print(f'data saved under the path {self.processed_data_path}/processed_data.csv')
        else:
            raise NotImplementedError("currently only pandas-based data preprocessing is supported")
                    
    def train_model(self, df):
        print('start training models soon...')

        if self.train_framework != 'spark':
            from src.training.pandas.model_training import Trainer
            trainer = Trainer(self.train_data_spec, df, self.train_model_spec, self.test_backend, self.in_memory, self.tmp_path, self.worker_ips, self.ray_params, self.hpo_spec)

            if self.hpo_needed:
                trainer.run_hpo()
            else:
                trainer.process()
                trainer.save_model(self.model_save_path)
        else:
            raise NotImplementedError('currently only pandas-based model training is supported')
        

    def clean_up(self):
        if self.cluster_engine is None:
            pass 
        elif self.cluster_engine == 'ray':
            import ray 
            ray.shutdown()
        else:
            raise NotImplementedError('spark engine is to be implemented!')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--config-file",
            required=True,
            type=str,
            help="speficy the config file name")
    parser.add_argument(
            "--mode",
            required=True,
            type=int,
            help="use 1 for docker, 0 for bare-metal")
    
    args, _ = parser.parse_known_args()
    wf_processor = WFProcessor(args.config_file, args.mode)
    
    start = time.time()
    wf_processor.prepare_env()
    print("prepare env took %.1f seconds" % ((time.time()-start)))

    if wf_processor.has_dp:
        dp_start = time.time()
        start = time.time()
        wf_processor.read_raw_data()
        print("dp read data took %.1f seconds" % ((time.time()-start)))
        start = time.time()
        wf_processor.pre_splitting_transform()
        print("dp transform pre-splitting data took %.1f seconds" % ((time.time()-start)))
        start = time.time()
        wf_processor.split_data()
        print("dp split data took %.1f seconds" % ((time.time()-start)))
        start = time.time()
        wf_processor.post_splitting_transform()
        print("dp transform post-splitting data took %.1f seconds" % ((time.time()-start)))
        start = time.time()
        wf_processor.save_processed_data()
        print("dp save data took %.1f seconds" % ((time.time()-start)))
        print("data preprocessing took %.1f seconds" % ((time.time()-dp_start)))
    if wf_processor.has_training:
        train_start = time.time()
        wf_processor.read_train_data()
        wf_processor.train_model(wf_processor.data)
        print("training took %.1f seconds" % ((time.time()-train_start)))
    if wf_processor.in_memory: 
        start = time.time()
        wf_processor.read_raw_data()
        wf_processor.pre_splitting_transform()
        wf_processor.split_data()
        wf_processor.post_splitting_transform()
        if wf_processor.dp_engine == 'modin':
            import modin.pandas as pd 
        elif wf_processor.dp_engine == 'pandas':
            import pandas as pd  
        else:
            raise ImportError("currently only pandas-based data preprocessing is supported")

        df = pd.concat([wf_processor.train_data, wf_processor.test_data])
        wf_processor.train_model(df)
        print("end2end training took %.1f seconds" % ((time.time()-start)))

    wf_processor.clean_up()
    print('The whole workflow processing took %.1f seconds'%(time.time()-very_start))
