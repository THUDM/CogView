# -*- encoding: utf-8 -*-
'''
@File    :   preprocess_text_image_data.py
@Time    :   2021/01/24 15:38:44
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
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
from .pretokenized_data import make_text_image_batch, make_tuple_text_image_batch, make_super_resolution_batch
import PIL
import timeit



@torch.no_grad()
def extract_code(model, datasets, text_dict, name, device, txt_type):
    index = 0
    map_size = 1024 * 1024 * 1024 * 1024
    lmdb_env = lmdb.open(f'/root/mnt/lmdb/{name}', map_size=map_size, writemap=True)
    print(f'/root/mnt/lmdb/{name}')
    with lmdb_env.begin(write=True) as txn:
        for dataset in datasets:
            loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)
            print(dataset)
            pbar = tqdm(loader)
            for raw_imgs, raw_filenames in pbar:
                imgs = []
                filenames = []
                for i, filename in enumerate(raw_filenames):
                    if filename != "not_a_image" and text_dict.__contains__(filename):
                        imgs.append(raw_imgs[i])
                        filenames.append(filename)
                    else:
                        print("warning: deleted damaged image")
                imgs = torch.stack(imgs)
                imgs = imgs.to(device)
                try:
                    if txt_type == "h5":
                        filenames = filenames.numpy()
                    txts = [text_dict[filename] for filename in filenames]
                    if txt_type != "h5":
                        codes = make_text_image_batch(model, txts, imgs)
                    else:
                        codes = make_tuple_text_image_batch(model, txts, imgs)
                    for code in codes:
                        txn.put(str(index).encode('utf-8'), pickle.dumps(code))
                        index += 1
                except KeyError:
                    print("warning: KeyError. The text cannot be find")
                    pass
        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


@torch.no_grad()
def extract_code_super_resolution_patches(model, datasets, text_dict, name, device, txt_type):
    index = 0
    map_size = 1024 * 1024 * 1024 * 1024
    lmdb_env = lmdb.open(f'/root/mnt/lmdb/{name}_super_resolution', map_size=map_size, writemap=True)
    print(f'/root/mnt/lmdb/{name}_super_resolution')
    with lmdb_env.begin(write=True) as txn:
        for dataset in datasets:
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
            print(dataset)
            pbar = tqdm(loader)
            for raw_imgs, raw_filenames in pbar:
                imgs = []
                filenames = []
                for i, filename in enumerate(raw_filenames):
                    if filename != "not_a_image" and text_dict.__contains__(filename):
                        imgs.append(raw_imgs[i])
                        filenames.append(filename)
                    else:
                        print("warning: deleted damaged image")
                imgs = torch.stack(imgs)
                imgs = imgs.to(device)
                try:
                    if txt_type == "h5":
                        filenames = filenames.numpy()
                    txts = [text_dict[filename] for filename in filenames]
                    if txt_type != "h5":
                        codes = make_super_resolution_batch(model, txts, imgs)
                    else:
                        codes = make_tuple_text_image_batch(model, txts, imgs)
                    for code in codes:
                        txn.put(str(index).encode('utf-8'), pickle.dumps(code))
                        index += 1
                except KeyError:
                    print("warning: KeyError. The text cannot be find")
                    pass
        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))

