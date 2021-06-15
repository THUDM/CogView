# -*- encoding: utf-8 -*-
'''
@File    :   preprocess_text_jsonformat_data.py
@Time    :   2021/03/14 20:56:28
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from .pretokenized_data import make_cut_text_batch
import timeit
import ujson as json

def extract_code(datasets, name, seq_len):
    '''
        datasets: [json_name1, json_name2, ...]
    '''
    index = 0
    map_size = 1024 * 1024 * 1024 * 1024
    lmdb_env = lmdb.open(f'/root/mnt/lmdb/{name}', map_size=map_size, writemap=True)
    with lmdb_env.begin(write=True) as txn:
        for dataset in datasets:
            with open(dataset, 'r') as fin:
                print(f'Loading {dataset}...')
                raw_json = json.load(fin)["RECORDS"]
            bs = 512
            for i in tqdm(range(0, len(raw_json), bs)):
                txts = [t["content"] for t in raw_json[i: i + bs]]
                txts = make_cut_text_batch(txts, seq_len)
                for code in txts:
                    txn.put(str(index).encode('utf-8'), pickle.dumps(code))
                    index += 1
        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))
    print(f'/root/mnt/lmdb/{name}, length={index}')
