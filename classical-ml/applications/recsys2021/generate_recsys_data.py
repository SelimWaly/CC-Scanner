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

import numpy as np
import random
import pandas as pd 
import argparse
import pathlib
import os 

class DataGenerator:
    
    def __init__(self, schema, seed=42):
        self.schema = schema 
        np.random.seed(seed)
        random.seed(seed)

    def generate_hashtags(self, num_rows_per_file):
        return [self.generate_hashtag() for _ in range(num_rows_per_file)]
    
    def generate_hashtag(self):
        length = np.random.poisson(1)
        if length == 0:
            return None
        else:
            text_tags = [self.generate_random_uuid() for _ in range(length)]
            return '\t'.join(text_tags)
        
    def generate_random_uuids(self, num_rows_per_file):
        return [ self.generate_random_uuid() for _ in range(num_rows_per_file)]
        
    def generate_random_uuid(self):        
        return ('%032x' % random.randrange(16**32)).upper()
             
    def generate_random_tokens(self, num_rows_per_file):
        return [ self.generate_random_token() for _ in range(num_rows_per_file)]
    
    def generate_random_token(self):
        tokens = ['101']
        tokens.extend(np.random.randint(100, 99998, np.random.randint(1, 80)).astype(str))
        tokens.append('102')
        return '\t'.join(tokens)
    
    def generate_random_medias(self, num_rows_per_file):
        return [self.generate_random_media() for _ in range(num_rows_per_file)]
        
    def generate_random_media(self):
        length = np.random.randint(4)
        if length == 0:
            return None
        else:
            media = [np.random.choice(['Photo', 'Video', 'GIF']) for _ in range(length)]
            return '\t'.join(media)
            
    def generate_follower_counts(self, num_rows_per_file, mu, sigma):
        
        return np.rint(abs(np.random.normal(loc=mu, scale=sigma,size=num_rows_per_file))).astype(int)
    
    def generate_engage_timestamp(self, num_rows_per_file, ratio):
        
        n = int(ratio*num_rows_per_file)
        data = np.random.randint(1612108800, 1614441600, num_rows_per_file).astype(float)
        index_nan = np.random.choice(data.size, n, replace=False)
        
        data.ravel()[index_nan] = np.NaN
        
        return data 
    
    def process(self, num_rows, num_files, save_path):
        
        num_rows_per_file = num_rows // num_files 
        
        for file in range(num_files):
            
            df = pd.DataFrame(self.schema)
            df['text_tokens'] = self.generate_random_tokens(num_rows_per_file)
            df['hashtags'] = self.generate_hashtags(num_rows_per_file)
            df['tweet_id'] = self.generate_random_uuids(num_rows_per_file)
            df['present_media'] = self.generate_random_medias(num_rows_per_file)
            df['present_links'] = self.generate_hashtags(num_rows_per_file)
            df['present_domains'] = self.generate_hashtags(num_rows_per_file)
            df['tweet_type'] = [np.random.choice(['Quote','TopLevel','Retweet']) for _ in range(num_rows_per_file)]
            df['language'] = self.generate_hashtags(num_rows_per_file)
            df['tweet_timestamp'] = np.random.randint(1612108800, 1614441600, num_rows_per_file)
            df['engaged_with_user_id'] = self.generate_hashtags(num_rows_per_file)
            df['engaged_with_user_follower_count'] = self.generate_follower_counts(num_rows_per_file, mu=857237, sigma=4737511)
            df['engaged_with_user_following_count'] = self.generate_follower_counts(num_rows_per_file, mu=383, sigma=329272)
            df['engaged_with_user_is_verified'] = np.random.choice([True,False], num_rows_per_file)
            df['engaged_with_user_account_creation'] = np.random.randint(1267286400, 1614441600, num_rows_per_file)
            df['engaging_user_id'] = self.generate_hashtags(num_rows_per_file)
            df['engaging_user_follower_count'] = self.generate_follower_counts(num_rows_per_file, mu=842, sigma=30586)
            df['enaging_user_following_count'] = self.generate_follower_counts(num_rows_per_file, mu=712, sigma=1681)
            df['enaging_user_is_verified'] = np.random.choice([True,False], num_rows_per_file)
            df['engaging_user_account_creation'] = np.random.randint(1267286400, 1614441600, num_rows_per_file)
            df['engagee_follows_engager'] = np.random.choice([True,False], num_rows_per_file)
            df['reply_timestamp'] = self.generate_engage_timestamp(num_rows_per_file, 0.026)
            df['retweet_timestamp'] = self.generate_engage_timestamp(num_rows_per_file, 0.088)
            df['retweet_with_comment_timestamp'] = self.generate_engage_timestamp(num_rows_per_file, 0.0067)
            df['like_timestamp'] = self.generate_engage_timestamp(num_rows_per_file, 0.4)
            df['tokens'] = df['text_tokens'].apply(lambda x: x.split('\t'))
            
            df.to_parquet(save_path+f'/part-{file}.parquet')


def check_data_path(save_path):
    path = pathlib.Path(save_path)
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    data_path = args.save_path
    data_path = os.path.abspath(data_path)
    save_train_path = os.path.join(data_path,'train')
    save_valid_path = os.path.join(data_path,'valid')

    check_data_path(save_train_path)
    check_data_path(save_valid_path)

    df_schema = df_schema = {
                'text_tokens': pd.Series(dtype='str'),
                'hashtags': pd.Series(dtype='str'),
                'tweet_id': pd.Series(dtype='str'),
                'present_media': pd.Series(dtype='str'),
                'present_links': pd.Series(dtype='str'),
                'present_domains': pd.Series(dtype='str'),
                'tweet_type': pd.Series(dtype='str'),
                'language': pd.Series(dtype='str'),
                'tweet_timestamp': pd.Series(dtype='int'),
                'engaged_with_user_id': pd.Series(dtype='str'),
                'engaged_with_user_follower_count': pd.Series(dtype='int'),
                'engaged_with_user_following_count': pd.Series(dtype='int'),
                'engaged_with_user_is_verified': pd.Series(dtype='bool'),
                'engaged_with_user_account_creation': pd.Series(dtype='int'),
                'engaging_user_id': pd.Series(dtype='str'),
                'engaging_user_follower_count': pd.Series(dtype='int'),
                'enaging_user_following_count': pd.Series(dtype='int'),
                'enaging_user_is_verified': pd.Series(dtype='bool'),
                'engaging_user_account_creation': pd.Series(dtype='int'),
                'engagee_follows_engager': pd.Series(dtype='bool'),
                'reply_timestamp': pd.Series(dtype='float'),
                'retweet_timestamp': pd.Series(dtype='float'),
                'retweet_with_comment_timestamp': pd.Series(dtype='float'),
                'like_timestamp': pd.Series(dtype='float'),
                'tokens': pd.Series(dtype='str')
            }

    dg = DataGenerator(df_schema)

    ## the total number of rows in training data should be an integral multiple of the file numbers, e.g. 51996000/200=259980    
    dg.process(51996000, 200, save_train_path)    
    print("Recsys2021 Training Data is generated!")
    dg.process(14461760, 1, save_valid_path)
    print("Recsys2021 Validation Data is generated!")


