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

import gc
from timeit import default_timer as timer
import hashlib

def split_train(df, proc, output_name, sample_ratio=0.083):
    t1 = timer()
    df = df.sample(sample_ratio, seed=3)
    gc.collect()
    df.write.format('parquet').mode('overwrite').save(proc.path_prefix + proc.current_path + output_name)
    t2 = timer()
    print("select train took %.3f seconds" % (t2 - t1))
    return df

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