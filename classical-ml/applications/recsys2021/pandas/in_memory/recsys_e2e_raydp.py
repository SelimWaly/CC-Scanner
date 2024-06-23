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

import ray
import numpy as np
# XGBoost on ray is needed to run this example.
# Please refer to https://docs.ray.io/en/latest/xgboost-ray.html to install it.
from xgboost_ray import RayDMatrix, train, RayParams, predict, RayShardingMode
from sklearn.metrics import log_loss, average_precision_score
import raydp
import pyrecdp 
import gc,time,os
from RecsysSchema import RecsysSchema
from pyrecdp.data_processor import *
from pyrecdp.encoder import *
from pyrecdp.utils import *
from features import *
import hashlib

very_start = time.time()

XGB_PARAMS = { 
'max_depth':8, 
'learning_rate':0.1, 
'subsample':0.8,
'colsample_bytree':0.8, 
'eval_metric':'logloss',
'objective':'binary:logistic',
'tree_method':'hist',
"random_state":42
}

target_list = [
    'reply_timestamp',
    'retweet_timestamp',
    'retweet_with_comment_timestamp',
    'like_timestamp'
    ]
indexlist = ["engaging_user_id","tweet_id"]
stage1_features_list = [
    'engaged_with_user_follower_count',
    'engaged_with_user_following_count',
    'engaged_with_user_is_verified',
    'engaging_user_follower_count',
    'engaging_user_following_count',
    'engaging_user_is_verified',
    'has_photo',
    'has_video',
    'has_gif',
    'a_ff_rate',
    'b_ff_rate', 
    'dt_hour',
    'dt_dow',
    'has_mention',  
    'mentioned_bucket_id',    
    'mentioned_count',    
    'most_used_word_bucket_id',
    'second_used_word_bucket_id',
    'TE_tweet_type_reply_timestamp',
    'TE_tweet_type_retweet_timestamp',
    'TE_dt_dow_retweet_timestamp',
    'TE_most_used_word_bucket_id_reply_timestamp',
    'TE_most_used_word_bucket_id_retweet_timestamp',
    'TE_most_used_word_bucket_id_retweet_with_comment_timestamp',
    'TE_most_used_word_bucket_id_like_timestamp',
    'TE_second_used_word_bucket_id_reply_timestamp',
    'TE_second_used_word_bucket_id_retweet_timestamp',
    'TE_second_used_word_bucket_id_retweet_with_comment_timestamp',
    'TE_second_used_word_bucket_id_like_timestamp',
    'TE_mentioned_bucket_id_retweet_timestamp',
    'TE_mentioned_bucket_id_retweet_with_comment_timestamp',
    'TE_mentioned_bucket_id_like_timestamp',
    'TE_mentioned_bucket_id_reply_timestamp',
    'TE_language_reply_timestamp',
    'TE_language_retweet_timestamp',
    'TE_language_retweet_with_comment_timestamp',
    'TE_language_like_timestamp',
    'TE_mentioned_count_reply_timestamp',
    'TE_mentioned_count_retweet_timestamp',
    'TE_mentioned_count_retweet_with_comment_timestamp',
    'TE_mentioned_count_like_timestamp',
    'TE_engaged_with_user_id_reply_timestamp',
    'TE_engaged_with_user_id_retweet_timestamp',
    'TE_engaged_with_user_id_retweet_with_comment_timestamp',
    'TE_engaged_with_user_id_like_timestamp',
    'GTE_language_engaged_with_user_id_reply_timestamp',
    'GTE_language_engaged_with_user_id_retweet_timestamp',
    'GTE_language_engaged_with_user_id_retweet_with_comment_timestamp',
    'GTE_language_engaged_with_user_id_like_timestamp',
    'GTE_tweet_type_engaged_with_user_id_reply_timestamp',
    'GTE_tweet_type_engaged_with_user_id_retweet_timestamp',
    'GTE_tweet_type_engaged_with_user_id_retweet_with_comment_timestamp',
    'GTE_tweet_type_engaged_with_user_id_like_timestamp',
    'GTE_has_mention_engaging_user_id_reply_timestamp',
    'GTE_has_mention_engaging_user_id_retweet_timestamp',
    'GTE_has_mention_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_has_mention_engaging_user_id_like_timestamp',
    'GTE_mentioned_bucket_id_engaging_user_id_reply_timestamp',
    'GTE_mentioned_bucket_id_engaging_user_id_retweet_timestamp',
    'GTE_mentioned_bucket_id_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_mentioned_bucket_id_engaging_user_id_like_timestamp',
    'GTE_language_engaging_user_id_reply_timestamp',
    'GTE_language_engaging_user_id_retweet_timestamp',
    'GTE_language_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_language_engaging_user_id_like_timestamp',
    'GTE_tweet_type_engaging_user_id_reply_timestamp',
    'GTE_tweet_type_engaging_user_id_retweet_timestamp',
    'GTE_tweet_type_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_tweet_type_engaging_user_id_like_timestamp',
    'GTE_dt_dow_engaged_with_user_id_reply_timestamp',
    'GTE_dt_dow_engaged_with_user_id_retweet_timestamp',
    'GTE_dt_dow_engaged_with_user_id_retweet_with_comment_timestamp',
    'GTE_dt_dow_engaged_with_user_id_like_timestamp',
    'GTE_mentioned_count_engaging_user_id_reply_timestamp',
    'GTE_mentioned_count_engaging_user_id_retweet_timestamp',
    'GTE_mentioned_count_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_mentioned_count_engaging_user_id_like_timestamp',
    'GTE_dt_hour_engaged_with_user_id_reply_timestamp',
    'GTE_dt_hour_engaged_with_user_id_retweet_timestamp',
    'GTE_dt_hour_engaged_with_user_id_retweet_with_comment_timestamp',
    'GTE_dt_hour_engaged_with_user_id_like_timestamp',
    'GTE_dt_dow_engaging_user_id_reply_timestamp',
    'GTE_dt_dow_engaging_user_id_retweet_timestamp',
    'GTE_dt_dow_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_dt_dow_engaging_user_id_like_timestamp',
    'GTE_dt_hour_engaging_user_id_reply_timestamp',
    'GTE_dt_hour_engaging_user_id_retweet_timestamp',
    'GTE_dt_hour_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_dt_hour_engaging_user_id_like_timestamp',
    "engagee_follows_engager",
    "dt_minute",
    "len_domains",
    "len_hashtags",
    "len_links",
    "TE_tweet_type_retweet_with_comment_timestamp",
    "TE_tweet_type_like_timestamp",
    "GTE_engaged_with_user_id_engaging_user_id_reply_timestamp",
    "GTE_engaged_with_user_id_engaging_user_id_retweet_timestamp",
    "GTE_engaged_with_user_id_engaging_user_id_retweet_with_comment_timestamp",
    "GTE_engaged_with_user_id_engaging_user_id_like_timestamp",
    "GTE_engaged_with_user_id_language_tweet_type_reply_timestamp",
    "GTE_engaged_with_user_id_language_tweet_type_retweet_timestamp",
    "GTE_engaged_with_user_id_language_tweet_type_retweet_with_comment_timestamp",
    "GTE_engaged_with_user_id_language_tweet_type_like_timestamp",
    "GTE_engaging_user_id_language_tweet_type_reply_timestamp",
    "GTE_engaging_user_id_language_tweet_type_retweet_timestamp",
    "GTE_engaging_user_id_language_tweet_type_retweet_with_comment_timestamp",
    "GTE_engaging_user_id_language_tweet_type_like_timestamp",
    "TE_engaging_user_id_reply_timestamp",
    "TE_engaging_user_id_retweet_timestamp",
    "TE_engaging_user_id_retweet_with_comment_timestamp",
    "TE_engaging_user_id_like_timestamp",
    "GTE_language_tweet_type_present_media_reply_timestamp",
    "GTE_language_tweet_type_present_media_retweet_timestamp",
    "GTE_language_tweet_type_present_media_retweet_with_comment_timestamp",
    "GTE_language_tweet_type_present_media_like_timestamp",
    "TE_present_media_reply_timestamp",
    "TE_present_media_retweet_timestamp",
    "TE_present_media_retweet_with_comment_timestamp",
    "TE_present_media_like_timestamp",
    'TE_tw_word0_reply_timestamp', 
    'TE_tw_word0_retweet_timestamp', 
    'TE_tw_word0_retweet_with_comment_timestamp', 
    'TE_tw_word0_like_timestamp',
    "len_media",
    "ab_age_dff",
    "ab_age_rate",
    "ab_fing_rate",
    "ab_fer_rate",
    'GTE_engaging_user_is_verified_tweet_type_reply_timestamp', 
    'GTE_engaging_user_is_verified_tweet_type_retweet_timestamp', 
    'GTE_engaging_user_is_verified_tweet_type_retweet_with_comment_timestamp', 
    'GTE_engaging_user_is_verified_tweet_type_like_timestamp',
    'GTE_present_domains_language_engagee_follows_engager_tweet_type_present_media_engaged_with_user_is_verified_reply_timestamp', 
    'GTE_present_domains_language_engagee_follows_engager_tweet_type_present_media_engaged_with_user_is_verified_retweet_timestamp', 
    'GTE_present_domains_language_engagee_follows_engager_tweet_type_present_media_engaged_with_user_is_verified_retweet_with_comment_timestamp', 
    'GTE_present_domains_language_engagee_follows_engager_tweet_type_present_media_engaged_with_user_is_verified_like_timestamp',
    'GTE_present_media_tweet_type_language_engaged_with_user_is_verified_engaging_user_is_verified_engagee_follows_engager_reply_timestamp', 
    'GTE_present_media_tweet_type_language_engaged_with_user_is_verified_engaging_user_is_verified_engagee_follows_engager_retweet_timestamp', 
    'GTE_present_media_tweet_type_language_engaged_with_user_is_verified_engaging_user_is_verified_engagee_follows_engager_retweet_with_comment_timestamp', 
    'GTE_present_media_tweet_type_language_engaged_with_user_is_verified_engaging_user_is_verified_engagee_follows_engager_like_timestamp',
    'GTE_tw_original_user0_tweet_type_language_reply_timestamp', 
    'GTE_tw_original_user0_tweet_type_language_retweet_timestamp', 
    'GTE_tw_original_user0_tweet_type_language_retweet_with_comment_timestamp', 
    'GTE_tw_original_user0_tweet_type_language_like_timestamp',
    'GTE_tw_original_user1_tweet_type_language_reply_timestamp', 
    'GTE_tw_original_user1_tweet_type_language_retweet_timestamp', 
    'GTE_tw_original_user1_tweet_type_language_retweet_with_comment_timestamp', 
    'GTE_tw_original_user1_tweet_type_language_like_timestamp'
    ]
stage2_features_list = [
    'stage2_TE_engaged_with_user_id_reply_timestamp',
    'stage2_TE_engaged_with_user_id_retweet_timestamp',
    'stage2_TE_engaged_with_user_id_retweet_with_comment_timestamp',
    'stage2_TE_engaged_with_user_id_like_timestamp',
    'stage2_TE_language_reply_timestamp',
    'stage2_TE_language_retweet_timestamp',
    'stage2_TE_language_retweet_with_comment_timestamp',
    'stage2_TE_language_like_timestamp',
    'stage2_TE_dt_dow_retweet_timestamp',
    'stage2_TE_tweet_type_reply_timestamp',
    'stage2_TE_tweet_type_retweet_timestamp',
    'stage2_TE_most_used_word_bucket_id_reply_timestamp',
    'stage2_TE_most_used_word_bucket_id_retweet_timestamp',
    'stage2_TE_most_used_word_bucket_id_retweet_with_comment_timestamp',
    'stage2_TE_most_used_word_bucket_id_like_timestamp',
    'stage2_TE_second_used_word_bucket_id_reply_timestamp',
    'stage2_TE_second_used_word_bucket_id_retweet_timestamp',
    'stage2_TE_second_used_word_bucket_id_retweet_with_comment_timestamp',
    'stage2_TE_second_used_word_bucket_id_like_timestamp',
    'stage2_TE_mentioned_count_reply_timestamp',
    'stage2_TE_mentioned_count_retweet_timestamp',
    'stage2_TE_mentioned_count_retweet_with_comment_timestamp',
    'stage2_TE_mentioned_count_like_timestamp',
    'stage2_TE_mentioned_bucket_id_reply_timestamp',
    'stage2_TE_mentioned_bucket_id_retweet_timestamp',
    'stage2_TE_mentioned_bucket_id_retweet_with_comment_timestamp',
    'stage2_TE_mentioned_bucket_id_like_timestamp',
    'stage2_GTE_has_mention_engaging_user_id_reply_timestamp',
    'stage2_GTE_has_mention_engaging_user_id_retweet_timestamp',
    'stage2_GTE_has_mention_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_has_mention_engaging_user_id_like_timestamp',
    'stage2_GTE_mentioned_count_engaging_user_id_reply_timestamp',
    'stage2_GTE_mentioned_count_engaging_user_id_retweet_timestamp',
    'stage2_GTE_mentioned_count_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_mentioned_count_engaging_user_id_like_timestamp',
    'stage2_GTE_mentioned_bucket_id_engaging_user_id_reply_timestamp',
    'stage2_GTE_mentioned_bucket_id_engaging_user_id_retweet_timestamp',
    'stage2_GTE_mentioned_bucket_id_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_mentioned_bucket_id_engaging_user_id_like_timestamp',
    'stage2_GTE_language_engaged_with_user_id_reply_timestamp',
    'stage2_GTE_language_engaged_with_user_id_retweet_timestamp',
    'stage2_GTE_language_engaged_with_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_language_engaged_with_user_id_like_timestamp',
    'stage2_GTE_language_engaging_user_id_reply_timestamp',
    'stage2_GTE_language_engaging_user_id_retweet_timestamp',
    'stage2_GTE_language_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_language_engaging_user_id_like_timestamp',
    'stage2_GTE_dt_dow_engaged_with_user_id_reply_timestamp',
    'stage2_GTE_dt_dow_engaged_with_user_id_retweet_timestamp',
    'stage2_GTE_dt_dow_engaged_with_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_dt_dow_engaged_with_user_id_like_timestamp',
    'stage2_GTE_dt_dow_engaging_user_id_reply_timestamp',
    'stage2_GTE_dt_dow_engaging_user_id_retweet_timestamp',
    'stage2_GTE_dt_dow_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_dt_dow_engaging_user_id_like_timestamp',
    'stage2_GTE_dt_hour_engaged_with_user_id_reply_timestamp',
    'stage2_GTE_dt_hour_engaged_with_user_id_retweet_timestamp',
    'stage2_GTE_dt_hour_engaged_with_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_dt_hour_engaged_with_user_id_like_timestamp',
    'stage2_GTE_dt_hour_engaging_user_id_reply_timestamp',
    'stage2_GTE_dt_hour_engaging_user_id_retweet_timestamp',
    'stage2_GTE_dt_hour_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_dt_hour_engaging_user_id_like_timestamp',
    'stage2_GTE_tweet_type_engaged_with_user_id_reply_timestamp',
    'stage2_GTE_tweet_type_engaged_with_user_id_retweet_timestamp',
    'stage2_GTE_tweet_type_engaged_with_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_tweet_type_engaged_with_user_id_like_timestamp',
    'stage2_GTE_tweet_type_engaging_user_id_reply_timestamp',
    'stage2_GTE_tweet_type_engaging_user_id_retweet_timestamp',
    'stage2_GTE_tweet_type_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_tweet_type_engaging_user_id_like_timestamp',
    'stage2_GTE_engaged_with_user_id_engaging_user_id_reply_timestamp',
    'stage2_GTE_engaged_with_user_id_engaging_user_id_retweet_timestamp',
    'stage2_GTE_engaged_with_user_id_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_engaged_with_user_id_engaging_user_id_like_timestamp',
    'stage2_GTE_engaged_with_user_id_language_tweet_type_reply_timestamp',
    'stage2_GTE_engaged_with_user_id_language_tweet_type_retweet_timestamp',
    'stage2_GTE_engaged_with_user_id_language_tweet_type_retweet_with_comment_timestamp',
    'stage2_GTE_engaged_with_user_id_language_tweet_type_like_timestamp',
    'stage2_GTE_engaging_user_id_language_tweet_type_reply_timestamp',
    'stage2_GTE_engaging_user_id_language_tweet_type_retweet_timestamp',
    'stage2_GTE_engaging_user_id_language_tweet_type_retweet_with_comment_timestamp',
    'stage2_GTE_engaging_user_id_language_tweet_type_like_timestamp',
    'stage2_TE_engaging_user_id_reply_timestamp',
    'stage2_TE_engaging_user_id_retweet_timestamp',
    'stage2_TE_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_TE_engaging_user_id_like_timestamp',
    'stage2_GTE_language_tweet_type_present_media_reply_timestamp',
    'stage2_GTE_language_tweet_type_present_media_retweet_timestamp',
    'stage2_GTE_language_tweet_type_present_media_retweet_with_comment_timestamp',
    'stage2_GTE_language_tweet_type_present_media_like_timestamp',
    'stage2_TE_present_media_reply_timestamp',
    'stage2_TE_present_media_retweet_timestamp',
    'stage2_TE_present_media_retweet_with_comment_timestamp',
    'stage2_TE_present_media_like_timestamp',
    'stage2_TE_tw_word0_reply_timestamp',
    'stage2_TE_tw_word0_retweet_timestamp',
    'stage2_TE_tw_word0_retweet_with_comment_timestamp',
    'stage2_TE_tw_word0_like_timestamp',
    'stage2_TE_tweet_id_reply_timestamp',
    'stage2_TE_tweet_id_retweet_timestamp',
    'stage2_TE_tweet_id_retweet_with_comment_timestamp',
    'stage2_TE_tweet_id_like_timestamp',
    'stage2_CE_engaged_with_user_id',
    'stage2_CE_engaging_user_id',
    'stage2_CE_language',
    'stage2_CE_present_media',
    'stage2_CE_tw_word0',
    'stage2_CE_tweet_id',
    'stage2_GCE_engaged_with_user_id_language_tweet_type',
    'stage2_GCE_engaging_user_id_language_tweet_type',
    'stage2_GCE_language_tweet_type_present_media',
    'stage2_GCE_engaged_with_user_id_engaging_user_id']

final_feature_list_stage1 = target_list + indexlist + stage1_features_list
final_feature_list_stage2 = final_feature_list_stage1 + stage2_features_list

TE_col_features_stage1 = [
            'engaged_with_user_id',
            'language',
            'dt_dow',
            'tweet_type',
            'most_used_word_bucket_id',
            'second_used_word_bucket_id',
            'mentioned_count',
            'mentioned_bucket_id',
            ['has_mention', 'engaging_user_id'],
            ['mentioned_count', 'engaging_user_id'],
            ['mentioned_bucket_id', 'engaging_user_id'],
            ['language', 'engaged_with_user_id'],
            ['language', 'engaging_user_id'],
            ['dt_dow', 'engaged_with_user_id'],
            ['dt_dow', 'engaging_user_id'],
            ['dt_hour', 'engaged_with_user_id'],
            ['dt_hour', 'engaging_user_id'],
            ['tweet_type', 'engaged_with_user_id'],
            ['tweet_type', 'engaging_user_id'],
            ['engaged_with_user_id','engaging_user_id'],
            ['engaged_with_user_id','language','tweet_type'],
            ['engaging_user_id','language','tweet_type'],
            'engaging_user_id',
            ['language','tweet_type','present_media'],
            'present_media',
            'tw_word0',
            ['engaging_user_is_verified','tweet_type'],
            ['present_domains', 'language', 'engagee_follows_engager', 'tweet_type', 'present_media', 'engaged_with_user_is_verified'],
            ['present_media', 'tweet_type', 'language', 'engaged_with_user_is_verified', 'engaging_user_is_verified', 'engagee_follows_engager'],
            ['tw_original_user0', 'tweet_type', 'language'],
            ['tw_original_user1', 'tweet_type', 'language'],
    ]
TE_col_features_stage1_threshold = {
    "TE_engaged_with_user_id": 0,
    "TE_language": 0,
    "TE_dt_dow": 0,
    "TE_tweet_type": 0,
    "TE_most_used_word_bucket_id": 0,
    "TE_second_used_word_bucket_id": 0,
    "TE_mentioned_count": 0,
    "TE_mentioned_bucket_id": 0,
    "GTE_has_mention_engaging_user_id": 2,
    "GTE_mentioned_count_engaging_user_id": 2,
    "GTE_mentioned_bucket_id_engaging_user_id": 2,
    "GTE_language_engaged_with_user_id": 1,
    "GTE_language_engaging_user_id": 2,
    "GTE_dt_dow_engaged_with_user_id": 2,
    "GTE_dt_dow_engaging_user_id": 4,
    "GTE_dt_hour_engaged_with_user_id": 3,
    "GTE_dt_hour_engaging_user_id": 4,
    "GTE_tweet_type_engaged_with_user_id": 1,
    "GTE_tweet_type_engaging_user_id": 3,
    "GTE_engaged_with_user_id_engaging_user_id": 3,
    "GTE_engaged_with_user_id_language_tweet_type": 1,
    "GTE_engaging_user_id_language_tweet_type": 3,
    "TE_engaging_user_id": 1,
    "GTE_language_tweet_type_present_media": 0,
    "TE_present_media": 0,
    "TE_tw_word0": 0,
    "GTE_engaging_user_is_verified_tweet_type": 0,
    "GTE_present_domains_language_engagee_follows_engager_tweet_type_present_media_engaged_with_user_is_verified": 0,
    "GTE_present_media_tweet_type_language_engaged_with_user_is_verified_engaging_user_is_verified_engagee_follows_engager": 0,
    "GTE_tw_original_user0_tweet_type_language": 0,
    "GTE_tw_original_user1_tweet_type_language": 0}
TE_col_features_stage2 = [
            'engaged_with_user_id',
            'language',
            'dt_dow',
            'tweet_type',
            'most_used_word_bucket_id',
            'second_used_word_bucket_id',
            'mentioned_count',
            'mentioned_bucket_id',
            ['has_mention', 'engaging_user_id'],
            ['mentioned_count', 'engaging_user_id'],
            ['mentioned_bucket_id', 'engaging_user_id'],
            ['language', 'engaged_with_user_id'],
            ['language', 'engaging_user_id'],
            ['dt_dow', 'engaged_with_user_id'],
            ['dt_dow', 'engaging_user_id'],
            ['dt_hour', 'engaged_with_user_id'],
            ['dt_hour', 'engaging_user_id'],
            ['tweet_type', 'engaged_with_user_id'],
            ['tweet_type', 'engaging_user_id'],
            ['engaged_with_user_id','engaging_user_id'],
            ['engaged_with_user_id','language','tweet_type'],
            ['engaging_user_id','language','tweet_type'],
            'engaging_user_id',
            ['language','tweet_type','present_media'],
            'present_media',
            'tw_word0',
            'tweet_id'
    ]
TE_col_excludes = {'dt_dow': ['reply_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']}
CE_col_features = ['engaged_with_user_id', 
                'engaging_user_id',
                'language',
                'present_media',
                'tw_word0',
                'tweet_id',
                ['engaged_with_user_id','language','tweet_type'],
                ['engaging_user_id','language','tweet_type'],
                ['language','tweet_type','present_media'],
                ['engaged_with_user_id','engaging_user_id']
    ]

def extract_rt(x_org):
    x = x_org.lower().replace('[sep]', '').replace('\[cls\] rt @', '@')
    x = x.split('http')[0]
    x = x.rstrip()
    return(x)

def hashit(x):
    uhash = '0' if len(x)<=2 else x
    hash_object = hashlib.md5(uhash.encode('utf-8'))
    return int(hash_object.hexdigest(),16)%2**32

def ret_word( x, rw=0 ):
    x = x.split(' ')

    if len(x)>rw:
        return hashit(x[rw])
    elif rw<0:
        if len(x)>0:
            return hashit(x[-1])
        else:
            return 0
    else:
        return 0
    
def extract_hash(text, split_text='@', no=0):
    text = text.lower()
    uhash = ''
    text_split = text.split('@')
    if len(text_split)>(no+1):
        text_split = text_split[no+1].split(' ')
        cl_loop = True
        uhash += clean_text(text_split[0])
        while cl_loop:
            if len(text_split)>1:
                if text_split[1] in ['_']:
                    uhash += clean_text(text_split[1]) + clean_text(text_split[2])
                    text_split = text_split[2:]
                else:
                    cl_loop = False
            else:
                cl_loop = False

    return hashit(uhash)

def clean_text(text):
    if len(text)>1:
        if text[-1] in ['!', '?', ':', ';', '.', ',']:
            return(text[:-1])
    return(text)

def check_last_char_quest(x_org):
    if len(x_org)<1:
        return(0)
    x = x_org.replace('[sep]', '')
    x = x.split('http')[0]
    if '#' in x:
        x = x.split('#')[0] + ' '.join(x.split('#')[1].split(' ')[1:])
    if '@' in x:
        x = x.split('@')[0] + ' '.join(x.split('@')[1].split(' ')[1:])
    x = x.rstrip()
    if len(x)<2:
        return(0)
    elif x[-1]=='?' and x[-2]!='!':
        return(1)
    elif x[-1]=='?' and x[-2]=='!':
        return(2)
    elif x[-1]=='!' and x[-2]=='?':
        return(3)
    elif x[-1]=='!' and x[-2]!='?':
        return(4)
    else:
        return(0)
    
def decodeBertTokenizerAndExtractFeatures(df, proc, output_name):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased', do_lower_case=False)

    op_fillna = FillNA(["present_domains","hashtags","present_links","present_media","tweet_type"],'')
    
    # define UDF
    tokenizer_decode = f.udf(lambda x: tokenizer.decode(
        [int(n) for n in x.split('\t')]))
    format_url = f.udf(lambda x: x.replace(
        'https : / / t. co / ', 'https://t.co/').replace('@ ', '@'))
    count_media = f.udf(lambda x: x.count('\t')+1 if x != '' else 0, t.IntegerType())
    udf_tw_word0 = f.udf(lambda x: ret_word(x,0), t.IntegerType())
    udf_tw_original_user0 = f.udf(lambda x: extract_hash(x, no=0))
    udf_tw_original_user1 = f.udf(lambda x: extract_hash(x, no=1))
    
    # define decode udf operations
    op_feature_modification_tokenizer_decode = FeatureAdd(
        cols={'tweet': 'text_tokens'}, udfImpl=tokenizer_decode)
    op_feature_modification_format_url = FeatureModification(
        cols=['tweet'], udfImpl=format_url)
    op_count_media = FeatureAdd(
        cols={'len_media': 'present_media'}, udfImpl=count_media)
    op_tw_word0 = FeatureAdd(
        cols={"tw_word0":"tweet"}, udfImpl=udf_tw_word0)
    op_tw_original_user0 = FeatureAdd(
        cols={'tw_original_user0': 'tweet'}, udfImpl=udf_tw_original_user0)
    op_tw_original_user1 = FeatureAdd(
        cols={'tw_original_user1': 'tweet'}, udfImpl=udf_tw_original_user1)
    
    op_feature_target_classify = FeatureModification(cols={
        "reply_timestamp": "f.when(f.col('reply_timestamp') > 0, 1).otherwise(0)",
        "retweet_timestamp": "f.when(f.col('retweet_timestamp') > 0, 1).otherwise(0)",
        "retweet_with_comment_timestamp": "f.when(f.col('retweet_with_comment_timestamp') > 0, 1).otherwise(0)",
        "like_timestamp": "f.when(f.col('like_timestamp') > 0, 1).otherwise(0)"}, op='inline')
    op_feature_dtype = FeatureModification(cols={
        "engagee_follows_engager": "f.col('engagee_follows_engager').cast(t.IntegerType())"
       }, op='inline')
    
    # define new features
    op_feature_from_original = FeatureAdd(
        cols={"has_photo": "f.col('present_media').contains('Photo').cast(t.IntegerType())",
              "has_video": "f.col('present_media').contains('Vedio').cast(t.IntegerType())",
              "has_gif": "f.col('present_media').contains('GIF').cast(t.IntegerType())",             
              "a_ff_rate": "f.col('engaged_with_user_following_count')/f.col('engaged_with_user_follower_count')",
              "b_ff_rate": "f.col('engaging_user_following_count') /f.col('engaging_user_follower_count')",
              "dt_dow": "f.dayofweek(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())",
              "dt_hour": "f.hour(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())",     
              "dt_minute": "f.minute(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())",     
              "mention": "f.regexp_extract(f.col('tweet'), r'[^RT]\s@(\S+)', 1)",
              "has_mention": "(f.col('mention')!= '').cast(t.IntegerType())",
              "len_domains":"f.when(f.col('present_domains') == '', f.lit(0)).otherwise(f.size(f.split(f.col('present_domains'), '\t'))).cast(t.IntegerType())",
              "len_hashtags": "f.when(f.col('hashtags') == '', f.lit(0)).otherwise(f.size(f.split(f.col('hashtags'), '\t'))).cast(t.IntegerType())",
              "len_links": "f.when(f.col('present_links') == '', f.lit(0)).otherwise(f.size(f.split(f.col('present_links'), '\t'))).cast(t.IntegerType())",
              "ab_age_dff":"f.col('engaged_with_user_account_creation') - f.col('engaging_user_account_creation')",
              "ab_age_rate":"(f.col('engaged_with_user_account_creation')+129)/(f.col('engaging_user_account_creation')+129)",
              "ab_fing_rate":"f.col('engaged_with_user_following_count')/(1+f.col('engaging_user_following_count'))",
              "ab_fer_rate":"f.col('engaged_with_user_follower_count')/(1+f.col('engaging_user_follower_count'))",
        }, op='inline')
    
    # execute
    proc.reset_ops([op_fillna,
                    op_feature_modification_tokenizer_decode,
                    op_feature_modification_format_url,
                    op_count_media,
                    op_tw_word0,
                    op_tw_original_user0,
                    op_tw_original_user1,
                    op_feature_target_classify,
                    op_feature_dtype,
                    op_feature_from_original])
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("BertTokenizer decode and feature extacting took %.3f" % (t2 - t1))

    return df

def categorifyFeatures(df, proc, output_name, gen_dict, sampleRatio=1):
    # 1. prepare dictionary
    dict_dfs = []
    if gen_dict:
        # only call below function when target dicts were not pre-prepared
        op_gen_dict_multiItems = GenerateDictionary(['tweet'], doSplit=True, sep=' ', bucketSize=100)
        op_gen_dict_singleItems = GenerateDictionary(['mention'], bucketSize=100)
        proc.reset_ops([op_gen_dict_multiItems, op_gen_dict_singleItems])
        t1 = timer()
        dict_dfs = proc.generate_dicts(df)
        t2 = timer()
        print("Generate Dictionary took %.3f" % (t2 - t1))
    else:
        dict_names = ['tweet', 'mention']
        dict_dfs = [{'col_name': name, 'dict': proc.spark.read.parquet(
            "%s/%s/%s/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, name))} for name in dict_names]
    # 2. since we need both mentioned_bucket_id and mentioned_count, add two mention id dict_dfs
    for dict_df in dict_dfs:
        if dict_df['col_name'] == 'mention':
            dict_dfs.append({'col_name': 'mentioned_bucket_id', 'dict':dict_df['dict']})
            dict_dfs.append({'col_name': 'mentioned_count', 'dict':dict_df['dict'].drop('dict_col_id').withColumnRenamed('count', 'dict_col_id')})
    op_feature_add = FeatureAdd({"mentioned_bucket_id": "f.col('mention')", "mentioned_count": "f.col('mention')"}, op='inline')
    
    # 3. categorify
    op_categorify_multiItems = Categorify([{'bucketized_tweet_word': 'tweet'}], dict_dfs=dict_dfs, doSplit=True, sep=' ')
    op_categorify_singleItem = Categorify(['mentioned_bucket_id', 'mentioned_count'], dict_dfs=dict_dfs)
    proc.reset_ops([op_feature_add, op_categorify_multiItems, op_categorify_singleItem])
    
    # 4. get most and second used bucketized_tweet_word
    op_feature_add_sorted_bucketized_tweet_word = FeatureAdd(
        cols={'sorted_bucketized_tweet_word': "f.expr('sortIntArrayByFrequency(bucketized_tweet_word)')"}, op='inline')
    op_feature_add_convert = FeatureAdd(
        cols={'most_used_word_bucket_id': "f.when(f.size(f.col('sorted_bucketized_tweet_word'))>0, f.col('sorted_bucketized_tweet_word').getItem(0)).otherwise(np.nan)",
             'second_used_word_bucket_id': "f.when(f.size(f.col('sorted_bucketized_tweet_word'))>1, f.col('sorted_bucketized_tweet_word').getItem(1)).otherwise(np.nan)"}, op='inline')
    proc.append_ops([op_feature_add_sorted_bucketized_tweet_word, op_feature_add_convert])

    # 5. transform
    t1 = timer()
    if sampleRatio != 1:
        df = df.sample(sampleRatio)
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("categorify and getMostAndSecondUsedWordBucketId took %.3f" % (t2 - t1))
    return (df, dict_dfs)

def CountEncodingFeatures(df, proc, gen_dict, mode, train_generate=True):
    if mode == 'stage1':
        features = CE_col_features
        prefix = ''
    elif mode == 'stage2':
        features = CE_col_features
        prefix = 'stage2_'
    elif mode == 'inference':
        features = CE_col_features
        prefix = "inference_"
    else:
        raise NotImplementedError("mode need to be train or valid")
    
    targets = ['reply_timestamp']

    t1 = timer()
    ce_train_dfs = []
    ce_test_dfs = []
    for c in features:
        target_tmp = targets
        out_name = ""
        out_col_list = []
        for tgt in target_tmp:
            if isinstance(c, list):
                out_col_list.append(prefix + 'GCE_'+'_'.join(c))
                out_name = prefix + 'GCE_'+'_'.join(c)
            else:
                out_col_list.append(prefix + f'CE_{c}')
                out_name = prefix + f'CE_{c}'
        if gen_dict:
            start = timer()
            encoder = CountEncoder(proc, c, target_tmp, out_col_list, out_name,train_generate=train_generate)
            if train_generate:
                ce_train_df, ce_test_df = encoder.transform(df)
                ce_train_dfs.append({'col_name': c, 'dict': ce_train_df})
            else:
                ce_test_df = encoder.transform(df)
            ce_test_dfs.append({'col_name': c, 'dict': ce_test_df})
            print(f"generating count encoding for %s upon %s took %.1f seconds"%(str(c), str(target_tmp), timer()-start))
        else:
            ce_train_path = "%s/%s/%s/train/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, out_name)
            ce_test_path = "%s/%s/%s/test/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, out_name)    
            if train_generate:           
                ce_train_dfs.append({'col_name': c, 'dict': proc.spark.read.parquet(ce_train_path)})
            ce_test_dfs.append({'col_name': c, 'dict': proc.spark.read.parquet(ce_test_path)})
    t2 = timer()
    print("Generate count encoding feature totally took %.3f" % (t2 - t1))

    if train_generate:
        return (ce_train_dfs, ce_test_dfs)
    else:
        return ce_test_dfs

def TargetEncodingFeatures(df, proc, gen_dict, mode):   
    if mode == 'stage1':
        features = TE_col_features_stage1
        prefix = ''
    elif mode == 'stage2':
        features = TE_col_features_stage2
        prefix = 'stage2_'
    else:
        raise NotImplementedError("mode need to be train or valid")

    targets = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    y_mean_all = []
    
    t1 = timer()
    if gen_dict:
        for tgt in targets:
            tmp = df.groupBy().mean(tgt).collect()[0]
            y_mean = tmp[f"avg({tgt})"]
            y_mean_all.append(y_mean)
        schema = t.StructType([t.StructField(tgt, t.FloatType(), True) for tgt in targets])
        y_mean_all_df = proc.spark.createDataFrame([tuple(y_mean_all)], schema)
        y_mean_all_df.write.format("parquet").mode("overwrite").save(
            "%s/%s/%s/targets_mean" % (proc.path_prefix, proc.current_path, proc.dicts_path))
    y_mean_all_df = proc.spark.read.parquet(
        "%s/%s/%s/targets_mean" % (proc.path_prefix, proc.current_path, proc.dicts_path))

    te_train_dfs = []
    te_test_dfs = []
    for c in features:
        target_tmp = targets
        out_name = ""
        if str(c) in TE_col_excludes:
            target_tmp = []
            for tgt in targets:
                if tgt not in TE_col_excludes[c]:
                    target_tmp.append(tgt)
        out_col_list = []
        for tgt in target_tmp:
            if isinstance(c, list):
                out_col_list.append(prefix + 'GTE_'+'_'.join(c)+'_'+tgt)
                out_name = prefix + 'GTE_'+'_'.join(c)
            else:
                out_col_list.append(prefix + f'TE_{c}_{tgt}')
                out_name = prefix + f'TE_{c}'
        if mode == "stage1":
            threshold = TE_col_features_stage1_threshold[out_name]
        else:
            threshold = 0
        if gen_dict:
            start = timer()
            encoder = TargetEncoder(proc, c, target_tmp, out_col_list, out_name, out_dtype=t.FloatType(), y_mean_list=y_mean_all,threshold=threshold)
            te_train_df, te_test_df = encoder.transform(df)
            te_train_dfs.append({'col_name': ['fold'] + (c if isinstance(c, list) else [c]), 'dict': te_train_df})
            te_test_dfs.append({'col_name': c, 'dict': te_test_df})
            print(f"generating target encoding for %s upon %s took %.1f seconds"%(str(c), str(target_tmp), timer()-start))
        else:
            te_train_path = "%s/%s/%s/train/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, out_name)
            te_test_path = "%s/%s/%s/test/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, out_name)               
            te_train_dfs.append({'col_name': ['fold'] + (c if isinstance(c, list) else [c]), 'dict': proc.spark.read.parquet(te_train_path)})
            te_test_dfs.append({'col_name': c, 'dict': proc.spark.read.parquet(te_test_path)})
    t2 = timer()
    print("Generate encoding feature totally took %.3f" % (t2 - t1))

    return (te_train_dfs, te_test_dfs, y_mean_all_df)

def mergeCountEncodingFeatures(df, ce_train_dfs, proc, output_name):
    # merge dicts to original table
    op_merge_CE = ModelMerge(ce_train_dfs)
    proc.reset_ops([op_merge_CE])
    
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("Merge Count Encoding Features took %.3f" % (t2 - t1))
    
    return df

def mergeTargetEncodingFeatures(df, te_train_dfs, proc, output_name, mode):
    feature_list = final_feature_list_stage1 if mode == 'stage1' else final_feature_list_stage2

    # merge dicts to original table
    op_merge_to_train = ModelMerge(te_train_dfs)
    proc.reset_ops([op_merge_to_train])
    
    # select features
    op_select = SelectFeature(feature_list)
    proc.append_ops([op_select])

    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("Merge Target Encoding Features took %.3f" % (t2 - t1))
    
    return df

def getTargetEncodingFeaturesDicts(proc, mode, train_dict_load = True):
    if mode == 'stage1':
        features = TE_col_features_stage1
        prefix = ''
    elif mode == 'stage2':
        features = TE_col_features_stage2
        prefix = 'stage2_'
    else:
        raise NotImplementedError("mode need to be stage1 or stage2")

    targets = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    y_mean_all = []
    y_mean_all_df = proc.spark.read.parquet(
        "%s/%s/%s/targets_mean" % (proc.path_prefix, proc.current_path, proc.dicts_path))

    te_train_dfs = []
    te_test_dfs = []
    for c in features:
        target_tmp = targets
        out_name = ""
        if str(c) in TE_col_excludes:
            target_tmp = []
            for tgt in targets:
                if tgt not in TE_col_excludes[c]:
                    target_tmp.append(tgt)
        for tgt in target_tmp:
            if isinstance(c, list):
                out_name = prefix + 'GTE_'+'_'.join(c)
            else:
                out_name = prefix + f'TE_{c}'
        if train_dict_load:
            te_train_path = "%s/%s/%s/train/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, out_name)
            te_train_dfs.append({'col_name': ['fold'] + (c if isinstance(c, list) else [c]), 'dict': proc.spark.read.parquet(te_train_path)})
        te_test_path = "%s/%s/%s/test/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, out_name)
        te_test_dfs.append({'col_name': c, 'dict': proc.spark.read.parquet(te_test_path)})
        
    return (te_train_dfs, te_test_dfs, y_mean_all_df)

def valid_mergeFeatures(df, te_test_dfs, y_mean_all_df, proc, output_name, mode, dict_dfs=None):
    proc.reset_ops([])
    if mode == "stage1":
        # categorify new data with train generated dictionary
        for dict_df in dict_dfs:
            if dict_df['col_name'] == 'mention':
                dict_dfs.append({'col_name': 'mentioned_bucket_id', 'dict':dict_df['dict']})
                dict_dfs.append({'col_name': 'mentioned_count', 'dict':dict_df['dict'].drop('dict_col_id').withColumnRenamed('count', 'dict_col_id')})
        op_feature_add = FeatureAdd({"mentioned_bucket_id": "f.col('mention')", "mentioned_count": "f.col('mention')"}, op='inline')
        
        op_categorify_multiItems = Categorify([{'bucketized_tweet_word': 'tweet'}], dict_dfs=dict_dfs, doSplit=True, sep=' ')
        op_categorify_singleItem = Categorify(['mentioned_bucket_id', 'mentioned_count'], dict_dfs=dict_dfs)
        proc.reset_ops([op_feature_add, op_categorify_multiItems, op_categorify_singleItem])
        
        op_feature_add_sorted_bucketized_tweet_word = FeatureAdd(
            cols={'sorted_bucketized_tweet_word': "f.expr('sortIntArrayByFrequency(bucketized_tweet_word)')"}, op='inline')
        op_feature_add_convert = FeatureAdd(
            cols={'most_used_word_bucket_id': "f.when(f.size(f.col('sorted_bucketized_tweet_word'))>0, f.col('sorted_bucketized_tweet_word').getItem(0)).otherwise(np.nan)",
                'second_used_word_bucket_id': "f.when(f.size(f.col('sorted_bucketized_tweet_word'))>1, f.col('sorted_bucketized_tweet_word').getItem(1)).otherwise(np.nan)"}, op='inline')
        proc.append_ops([op_feature_add_sorted_bucketized_tweet_word, op_feature_add_convert])
    
    # merge target encoding dicts 
    op_merge_to_test = ModelMerge(te_test_dfs)
    proc.append_ops([op_merge_to_test])
        
    y_mean_all = y_mean_all_df.collect()[0]
    te_feature_list = stage1_features_list if mode == "stage1" else stage2_features_list
    for tgt in target_list:
        to_fill_list = []
        for feature in te_feature_list:
            if 'TE_' in feature and tgt in feature:
                to_fill_list.append(feature)
        op_fill_na = FillNA(to_fill_list, y_mean_all[tgt])
        proc.append_ops([op_fill_na])
    
    # select features
    feature_list = final_feature_list_stage1 if mode == "stage1" else final_feature_list_stage2
    op_select = SelectFeature(feature_list)

    if mode == "stage1":
        t1 = timer()
        df = proc.transform(df, name=output_name+"_all")
        t2 = timer()
        print("valid_mergeFeatures for stage1 step1 took %.3f" % (t2 - t1))

        proc.reset_ops([op_select])

        t1 = timer()
        df = proc.transform(df, name=output_name)
        t2 = timer()
        print("valid_mergeFeatures for stage1 step2 took %.3f" % (t2 - t1))
    elif mode == "stage2":
        proc.append_ops([op_select])

        t1 = timer()
        df = proc.transform(df, name=output_name)
        t2 = timer()
        print("valid_mergeFeatures for stage2 took %.3f" % (t2 - t1))
    else:
        raise NotImplementedError("mode need to be stage1 or stage2")
    
    return df 

def inference_mergeFeatures(df, dict_dfs, ce_test_dfs,te_test_dfs, te_test_dfs_stage2, y_mean_all_df, y_mean_all_df_stage2, proc, output_name):
    ################ categorify test data with train generated dictionary
    # 1. since we need both mentioned_bucket_id and mentioned_count, add two mention id dict_dfs
    for dict_df in dict_dfs:
        if dict_df['col_name'] == 'mention':
            dict_dfs.append({'col_name': 'mentioned_bucket_id', 'dict':dict_df['dict']})
            dict_dfs.append({'col_name': 'mentioned_count', 'dict':dict_df['dict'].drop('dict_col_id').withColumnRenamed('count', 'dict_col_id')})
    op_feature_add = FeatureAdd({"mentioned_bucket_id": "f.col('mention')", "mentioned_count": "f.col('mention')"}, op='inline')
    
    # 2. categorify
    op_categorify_multiItems = Categorify([{'bucketized_tweet_word': 'tweet'}], dict_dfs=dict_dfs, doSplit=True, sep=' ',estimated_bytes=6)
    op_categorify_singleItem = Categorify(['mentioned_bucket_id', 'mentioned_count'], dict_dfs=dict_dfs,estimated_bytes=6)
    proc.reset_ops([op_feature_add, op_categorify_multiItems, op_categorify_singleItem])
    
    # 3. get most and second used bucketized_tweet_word
    op_feature_add_sorted_bucketized_tweet_word = FeatureAdd(
        cols={'sorted_bucketized_tweet_word': "f.expr('sortIntArrayByFrequency(bucketized_tweet_word)')"}, op='inline')
    op_feature_add_convert = FeatureAdd(
        cols={'most_used_word_bucket_id': "f.when(f.size(f.col('sorted_bucketized_tweet_word'))>0, f.col('sorted_bucketized_tweet_word').getItem(0)).otherwise(np.nan)",
             'second_used_word_bucket_id': "f.when(f.size(f.col('sorted_bucketized_tweet_word'))>1, f.col('sorted_bucketized_tweet_word').getItem(1)).otherwise(np.nan)"}, op='inline')
    proc.append_ops([op_feature_add_sorted_bucketized_tweet_word, op_feature_add_convert])
    
    ################ merge target encoding dicts from stage1
    op_merge_to_test = ModelMerge(te_test_dfs,estimated_bytes=6)
    proc.append_ops([op_merge_to_test])
        
    # set null in encoding features to y_mean
    y_mean_all = y_mean_all_df.collect()[0]
    te_feature_list = stage1_features_list 
    for tgt in target_list:
        to_fill_list = []
        for feature in te_feature_list:
            if 'TE_' in feature and tgt in feature:
                to_fill_list.append(feature)
        op_fill_na = FillNA(to_fill_list, y_mean_all[tgt])
        proc.append_ops([op_fill_na])
    
    ################ merge count encoding dicts 
    op_merge_CE = ModelMerge(ce_test_dfs)
    proc.append_ops([op_merge_CE])

    ################  merge target encoding dicts from stage2
    op_merge_to_test_2 = ModelMerge(te_test_dfs_stage2)
    proc.append_ops([op_merge_to_test_2])
        
    # set null in encoding features to y_mean
    y_mean_all_2 = y_mean_all_df_stage2.collect()[0]
    te_feature_list = stage2_features_list 
    for tgt in target_list:
        to_fill_list = []
        for feature in te_feature_list:
            if 'TE_' in feature and tgt in feature:
                to_fill_list.append(feature)
        op_fill_na = FillNA(to_fill_list, y_mean_all_2[tgt])
        proc.append_ops([op_fill_na])

    op_rename = FeatureAdd(cols={
        'stage2_CE_engaged_with_user_id':"f.col('inference_CE_engaged_with_user_id')",
        'stage2_CE_engaging_user_id':"f.col('inference_CE_engaging_user_id')",
        'stage2_CE_language':"f.col('inference_CE_language')",
        'stage2_CE_present_media':"f.col('inference_CE_present_media')",
        'stage2_CE_tw_word0':"f.col('inference_CE_tw_word0')",
        'stage2_CE_tweet_id':"f.col('inference_CE_tweet_id')",
        'stage2_GCE_engaged_with_user_id_language_tweet_type':"f.col('inference_GCE_engaged_with_user_id_language_tweet_type')",
        'stage2_GCE_engaging_user_id_language_tweet_type':"f.col('inference_GCE_engaging_user_id_language_tweet_type')",
        'stage2_GCE_language_tweet_type_present_media':"f.col('inference_GCE_language_tweet_type_present_media')",
        'stage2_GCE_engaged_with_user_id_engaging_user_id':"f.col('inference_GCE_engaged_with_user_id_engaging_user_id')"
       }, op='inline')
    proc.append_ops([op_rename])
    
    ################ select features
    feature_list = final_feature_list_stage2
    op_select = SelectFeature(feature_list)
    proc.append_ops([op_select])

    ################ perform transform
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("mergeFeaturesToTest  and save selected features took %.3f" % (t2 - t1))

    return df

def split_train(df, proc, output_name, sample_ratio=0.083):
    t1 = timer()
    df = df.sample(sample_ratio, seed=3)
    gc.collect()
    df.write.format('parquet').mode('overwrite').save(proc.path_prefix + proc.current_path + output_name)
    t2 = timer()
    print("select train took %.3f seconds" % (t2 - t1))
    return df

def split_valid_byrandom(df, proc, output_name_train, output_name_valid, sample_ratio_valid = 0.2):
    t1 = timer()
    df_train,df_valid = df.randomSplit([1 - sample_ratio_valid, sample_ratio_valid],seed=20)
    df_train.write.format('parquet').mode('overwrite').save(proc.path_prefix + proc.current_path + output_name_train)
    df_valid.write.format('parquet').mode('overwrite').save(proc.path_prefix + proc.current_path + output_name_valid)
    t2 = timer()
    print("split valid took %.3f seconds" % (t2 - t1))
    return df_train, df_valid

def split_valid_byindex(df, proc, train_output, test_output):
    t1 = timer()
    train_df = df.where(f.col("is_train") == 1)
    train_df.write.format('parquet').mode('overwrite').save(proc.path_prefix + proc.current_path + train_output)
    t2 = timer()
    print("split to train took %.3f" % (t2 - t1))
    
    t1 = timer()
    test_df = df.where(f.col("is_train") == 0)
    test_df.write.format('parquet').mode('overwrite').save(proc.path_prefix + proc.current_path + test_output)
    t2 = timer()
    print("split to test took %.3f" % (t2 - t1))
    
    return (proc.spark.read.parquet(proc.path_prefix + proc.current_path + train_output),
            proc.spark.read.parquet(proc.path_prefix + proc.current_path + test_output))


def setup_local(path_prefix,current_path,dicts_folder):

        recsysSchema = RecsysSchema()
        recdp_path = pyrecdp.__path__[0]
        scala_udf_jars = recdp_path + "/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"

        num_executors = 8
        cores_per_executor = 16
        memory_per_executor = "60G"

        ## Initialize Spark Session 
        spark = raydp.init_spark(app_name="Recsys2021_data_process", num_executors=num_executors, executor_cores=cores_per_executor, executor_memory=memory_per_executor,
                        configs={"spark.driver.extraClassPath": f"{scala_udf_jars}",
                                "spark.executor.extraClassPath": f"{scala_udf_jars}",
                                "spark.cleaner.periodicGC.interval": "15min",
                                "spark.driver.memory": "300g",
                                "spark.driver.maxResultSize": "16g",
                                "spark.local.dir": "/mnt/tmp/spark",
                                "spark.sql.broadcastTimeout": "7200",
                                "spark.executorEnv.HF_DATASETS_OFFLINE": "1",
                                "spark.executorEnv.TRANSFORMERS_OFFLINE": "1",
                                "spark.sql.adaptive.enabled": "True",
                                "spark.sql.codegen.maxFields": "300",
                                "spark.sql.debug.maxToStringFields": "10000",
                                "spark.sql.execution.arrow.pyspark.enabled": "True"
                                })

        schema = recsysSchema.toStructType()

        proc = DataProcessor(spark, path_prefix,
                        current_path=current_path, dicts_path=dicts_folder, shuffle_disk_capacity="1500GB", spark_mode='local')

        return spark, schema, proc 


def setup_standalone(path_prefix,current_path,dicts_folder):
        recsysSchema = RecsysSchema()
        
        recdp_path = pyrecdp.__path__[0]
        
        scala_udf_jars = recdp_path + "/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"
        #320
        num_executors = 4
        cores_per_executor = 40
        memory_per_executor = "200G"

        ## Initialize Spark Session 
        spark = raydp.init_spark(app_name="Recsys2021_data_process", num_executors=num_executors, executor_cores=cores_per_executor, executor_memory=memory_per_executor,
                        configs={"spark.driver.extraClassPath": f"{scala_udf_jars}",
                                "spark.executor.extraClassPath": f"{scala_udf_jars}",
                                "spark.cleaner.periodicGC.interval": "15min",
                                "spark.driver.memory": "60g",
                                "spark.driver.maxResultSize": "16g",
                                "spark.local.dir": "/mnt/tmp/spark",
                                "spark.sql.broadcastTimeout": "7200",
                                "spark.executorEnv.HF_DATASETS_OFFLINE": "1",
                                "spark.executorEnv.TRANSFORMERS_OFFLINE": "1",
                                "spark.sql.adaptive.enabled": "True",
                                "spark.sql.codegen.maxFields": "300",
                                "spark.sql.debug.maxToStringFields": "10000",
                                "spark.executor.memoryOverhead": "20g",
                                "spark.sql.execution.arrow.pyspark.enabled": "True"
                               })
        
        schema = recsysSchema.toStructType()

        proc = DataProcessor(spark, path_prefix,
                                current_path=current_path, dicts_path=dicts_folder, shuffle_disk_capacity="1500GB",spark_mode='standalone')
    
        return spark, schema,proc


def read_train_data(data_path, spark):

        df = spark.read.parquet(data_path)
        df = df.withColumnRenamed('enaging_user_following_count', 'engaging_user_following_count')
        df = df.withColumnRenamed('enaging_user_is_verified', 'engaging_user_is_verified')
        df = df.drop("tokens")
        df = df.withColumn("fold", f.round(f.rand(seed=42)*(5-1)).cast("int"))
        gc.collect()
        print("data loaded!")

        return df 


def read_valid_data_stage1(valid_data_path, index_data_path, spark):
        
        recsysSchema = RecsysSchema()
        schema = recsysSchema.toStructType()

        df = spark.read.schema(schema).option('sep', '\x01').csv(valid_data_path)
        df_index = spark.read.parquet(index_data_path).select(["tweet_id","engaging_user_id","is_train"])

        df = df.join(df_index,["tweet_id","engaging_user_id"],"left")
        del df_index
        gc.collect()
        print("data loaded!")

        return df 


def read_valid_data_stage2(data_path, spark):

        df = spark.read.parquet(data_path)
        print("data loaded!")

        return df


def process_train_data(df, proc):

        df = decodeBertTokenizerAndExtractFeatures(df, proc, output_name="train1_decode")
        print("data decoded!")

        df, dict_dfs = categorifyFeatures(df, proc, output_name="train2_categorify", gen_dict=True)
        print("data categorified!")

        te_train_dfs, te_test_dfs, y_mean_all_df = TargetEncodingFeatures(df, proc, gen_dict=True, mode="stage1")
        print("data encoded!")

        print("before select:", df.count())
        df = split_train(df, proc, output_name="train3_select", sample_ratio=0.083)
        print("after select:", df.count())
        print("data selected!")

        df = mergeTargetEncodingFeatures(df, te_train_dfs, proc, output_name="stage1_train", mode="stage1")
        print("data merged!")

        return df 


def process_valid_data_stage1(df, proc):

        df = decodeBertTokenizerAndExtractFeatures(df, proc, output_name="valid1_decode")
        print("data decoded!")

        dict_names = ['tweet', 'mention']
        dict_dfs = [{'col_name': name, 'dict': spark.read.parquet(
                        "%s/%s/%s/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, name))} for name in dict_names]
        _, te_test_dfs, y_mean_all_df = getTargetEncodingFeaturesDicts(proc,mode='stage1')

        val_df = valid_mergeFeatures(df, te_test_dfs, y_mean_all_df, proc, output_name="stage1_valid",mode="stage1", dict_dfs=dict_dfs)
        print("val data merged!")

        return val_df 


def process_valid_data_stage2(df, proc):

        ce_train_dfs, ce_test_dfs = CountEncodingFeatures(df, proc, gen_dict=True, mode="stage2")
        print("count encoded!")

        df = mergeCountEncodingFeatures(df, ce_train_dfs, proc, output_name = "valid1_withCE")
        print("count encoding merged!")

        print("Before split:", df.count())
        df_train, df_valid = split_valid_byindex(df, proc, train_output="valid2_train", test_output="valid2_valid")
        print("after split, train:", df_train.count())
        print("after split, valid:", df_valid.count())
        print("split done!")

        numFolds = 5
        df_train = df_train.withColumn("fold", f.round(f.rand(seed=42)*(numFolds-1)).cast("int"))
        te_train_dfs, te_test_dfs, y_mean_all_df = TargetEncodingFeatures(df_train, proc, gen_dict=True, mode = 'stage2')
        print("target encoded!")

        df_train = mergeTargetEncodingFeatures(df_train, te_train_dfs, proc, output_name='stage2_train', mode='stage2')
        print("train target encoding merged!")

        df_valid = valid_mergeFeatures(df_valid, te_test_dfs, y_mean_all_df, proc, output_name="stage2_valid",mode='stage2')
        print("val data merged!")

        return (df_train, df_valid)   


def compute_AP(pred, gt):
    return average_precision_score(gt, pred)


def compute_rce_fast(pred, gt):
    cross_entropy = log_loss(gt, pred)
    yt = np.mean(gt)     
    strawman_cross_entropy = -(yt*np.log(yt) + (1 - yt)*np.log(1 - yt))
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0


def evaluate_results(oof, valid, label_names):
        print('#'*25);print('###','Evalution Results');print('#'*25)
        txts = ''
        sumap = 0
        sumrce = 0
        for i in range(4):
                ap = compute_AP(oof[:,i],valid[label_names[i]].values)
                rce = compute_rce_fast(oof[:,i],valid[label_names[i]].values)
                txt = f"{label_names[i]:20} AP:{ap:.5f} RCE:{rce:.5f}"
                print(txt)

                txts += "%.4f" % ap + ' '
                txts += "%.4f" % rce + ' '
                sumap += ap
                sumrce += rce
        print(txts)
        print("AVG AP: ", sumap/4.)
        print("AVG RCE: ", sumrce/4.)
        

def stage1_training(ray_train_df1, ray_valid_df1, model_save_path, ray_params):

        label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
        feature_list = []
        feature_list.append(stage1_reply_features)
        feature_list.append(stage1_retweet_features)
        feature_list.append(stage1_comment_features)
        feature_list.append(stage1_like_features)
        for i in range(4):
                print(len(feature_list[i])-1)

        oof = np.zeros((ray_valid_df1.count(),len(label_names)))

        for numlabel in range(4):
                name = label_names[numlabel]
                print('#'*25);print('###',name);print('#'*25)

                start = time.time()
                dtrain = RayDMatrix(ray_train_df1.select_columns(cols=feature_list[numlabel]), label=name)
                dvalid = RayDMatrix(ray_valid_df1.select_columns(cols=feature_list[numlabel]), label=name)
                print("prepare matrix took %.1f seconds" % ((time.time()-start)))

                print("Training.....")

                model = train(
                        XGB_PARAMS,
                        dtrain,
                        evals=[(dtrain, "train"), (dvalid, "eval")],
                        num_boost_round=500,
                        early_stopping_rounds=25,
                        verbose_eval=25,
                        ray_params=ray_params
                        )
                
                model.save_model(f"{model_save_path}/xgboost_{name}_stage1.model")

                dvalid = RayDMatrix(
                                ray_valid_df1.select_columns(cols=feature_list[numlabel]),
                                label=name,  # Will select this column as the label
                                sharding=RayShardingMode.BATCH
                                )

                oof[:, numlabel] = predict(model, dvalid, ray_params=ray_params)
        
        valid = ray_valid_df1.select_columns(cols=["tweet_id","engaging_user_id"]).to_pandas(limit=14461760)
        gt = ray_valid_df1.select_columns(cols=label_names).to_pandas(limit=14461760)

        for i in range(4):
                valid[f"pred_{label_names[i]}"] = oof[:,i]
        
        evaluate_results(oof, gt, label_names)

        return valid 


def data_merge(df1, df2, preds, data_path):

        index_cols = ['tweet_id', 'engaging_user_id']
        df1 = df1.merge(preds, on=index_cols, how="left")
        df2 = df2.merge(preds, on=index_cols, how="left")

        df1.to_parquet(f"{data_path}/stage2_train_pred.parquet")
        df2.to_parquet(f"{data_path}/stage2_valid_pred.parquet")
        return df1, df2 


def stage2_training(ray_train_df2, ray_valid_df2, model_save_path, ray_params):
        
        label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
        feature_list = []
        feature_list.append(stage2_reply_features)
        feature_list.append(stage2_retweet_features)
        feature_list.append(stage2_comment_features)
        feature_list.append(stage2_like_features)
        for i in range(4):
                print(len(feature_list[i])-1)

        oof = np.zeros((ray_valid_df2.count(),len(label_names)))

        for numlabel in range(4):
                name = label_names[numlabel]
                print('#'*25);print('###',name);print('#'*25)

                start = time.time()
                dtrain = RayDMatrix(ray_train_df2.select_columns(cols=feature_list[numlabel]), label=name)
                dvalid = RayDMatrix(ray_valid_df2.select_columns(cols=feature_list[numlabel]), label=name)
                print("prepare matrix took %.1f seconds" % ((time.time()-start)))

                print("Training.....")
                
                model = train(
                        XGB_PARAMS,
                        dtrain,
                        evals=[(dtrain, "train"), (dvalid, "eval")],
                        num_boost_round=500,
                        early_stopping_rounds=25,
                        verbose_eval=25,
                        ray_params=ray_params
                        )
                
                model.save_model(f"{model_save_path}/xgboost_{name}_stage2.model")

                dvalid = RayDMatrix(
                                ray_valid_df2.select_columns(cols=feature_list[numlabel]),
                                columns=feature_list[numlabel],
                                label=name,  # Will select this column as the label
                                sharding=RayShardingMode.BATCH
                                )

                oof[:, numlabel] = predict(model, dvalid, ray_params=ray_params)
                
        valid = ray_valid_df2.select_columns(cols=label_names).to_pandas(limit=14461760)

        evaluate_results(oof, valid, label_names)

        return None

if __name__ == "__main__":

        mode = sys.argv[1]
        MASTER = sys.argv[2]

        path_prefix = f"hdfs://{MASTER}:9000/"
        current_path = "/recsys2021/datapre_stage1/"
        train_folder = "/recsys2021/oridata/train/"
        valid_folder = "/recsys2021/oridata/valid/valid"
        original_index = '/recsys2021/oridata/valid/valid_split_index.parquet'
        dicts_folder = "recsys_dicts/"

        model_save_path = '/workspace/tmp/models/'
        data_path = '/workspace/tmp/data/'

        if mode == 'local':
                ray.init(address="local", num_cpus=160, _temp_dir="/workspace/tmp/ray")
                cpus_per_actor = 154
                num_actors = 1
                ray_params = RayParams(max_actor_restarts=2, num_actors=num_actors, cpus_per_actor=cpus_per_actor)
                spark, schema, proc = setup_local(path_prefix,current_path,dicts_folder)
        elif mode == 'standalone':
                print("start standalone....")
                ray.init(address='auto', _temp_dir="/workspcae/tmp/ray")
                cpus_per_actor = 15
                num_actors = 20
                ray_params = RayParams(num_actors=num_actors, cpus_per_actor=cpus_per_actor, elastic_training=True, max_failed_actors=1, max_actor_restarts=2)
                spark, schema, proc = setup_standalone(path_prefix,current_path,dicts_folder)
        else: 
                print("cluster mode should either be local or standalone")

        start = time.time()
        train_df = read_train_data(path_prefix + train_folder, spark)
        train_df = process_train_data(train_df, proc)
        print("processing train data took %.1f seconds" % ((time.time()-start)))

        start = time.time()
        valid_df = read_valid_data_stage1(path_prefix + valid_folder, path_prefix+original_index, spark)
        valid_df = process_valid_data_stage1(valid_df, proc)
        print("processing validation data for stage1 took %.1f seconds" % ((time.time()-start)))

        start = time.time()
        proc.current_path = "/recsys2021/datapre_stage2/"
        df = read_valid_data_stage2(path_prefix+"/recsys2021/datapre_stage1/stage1_valid_all", spark)
        stage2_train, stage2_valid =  process_valid_data_stage2(df, proc)
        print("processing validation data for stage2 took %.1f seconds" % ((time.time()-start)))

        print("converting data....")
        
        # train_df = spark.read.parquet(path_prefix+'/recsys2021/datapre_stage1/stage1_train/')
        # valid_df = spark.read.parquet(path_prefix+'/recsys2021/datapre_stage1/stage1_valid/')
        # stage2_train = spark.read.parquet(path_prefix+'/recsys2021/datapre_stage2/stage2_train/').toPandas()
        # stage2_valid = spark.read.parquet(path_prefix+'/recsys2021/datapre_stage2/stage2_valid/').toPandas()

        start = time.time()
        ray_train_df1 = raydp.spark.dataset.spark_dataframe_to_ray_dataset(train_df, _use_owner=True)
        ray_valid_df1 = raydp.spark.dataset.spark_dataframe_to_ray_dataset(valid_df, _use_owner=True)
        stage2_train = stage2_train.toPandas()
        stage2_valid = stage2_valid.toPandas()
        print("converting spark dataframe took %.1f seconds" % ((time.time()-start)))

        print(f"({ray_train_df1.count()}, {len(ray_train_df1.take(1)[0])})")
        print(f"({ray_valid_df1.count()}, {len(ray_valid_df1.take(1)[0])})")

        print(stage2_train.shape)
        print(stage2_valid.shape)

        raydp.stop_spark(cleanup_data=False)

        print("start stage1 training....")

        start = time.time()
        preds = stage1_training(ray_train_df1, ray_valid_df1, model_save_path, ray_params)
        print("stage1 training took %.1f seconds" % ((time.time()-start)))

        print("start data merging....")
        start = time.time()
        stage2_train, stage2_valid = data_merge(stage2_train, stage2_valid, preds, data_path)
        print("data merging took %.1f seconds" % ((time.time()-start)))
        
        print("converting from padnas to ray dataset....")
        ray_stage2_train = ray.data.from_pandas(stage2_train).repartition(20)
        ray_stage2_valid = ray.data.from_pandas(stage2_valid).repartition(20)
        print("converting data types took %.1f seconds" % ((time.time()-start)))

        print("start stage2 training....")
        start = time.time()
        stage2_training(ray_stage2_train, ray_stage2_valid, model_save_path, ray_params)
        print("stage2 training took %.1f seconds" % ((time.time()-start)))

        ray.shutdown()

        print('This notebook took %.1f seconds'%(time.time()-very_start))




