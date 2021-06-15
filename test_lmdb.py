import lmdb
import os, sys
from data_utils import get_tokenizer

def initialize(file_name):
    env = lmdb.open(file_name, "r")
    return env

def insert(env, sid, name):
    txn = env.begin(write=True)
    txn.put(str(sid).encode('utf-8'), name.encode('utf-8'))
    txn.commit()

def delete(env, sid):
    txn = env.begin(write=True)
    txn.delete(str(sid).encode('utf-8'))
    txn.commit()

def update(env, sid, name):
    txn = env.begin(write=True)
    txn.put(str(sid).encode('utf-8'), name.encode('utf-8'))
    txn.commit()


import pickle
def search(env, sid):
    txn = env.begin()
    data = pickle.loads(txn.get(str(sid).encode('utf-8')))
    return data

import argparse
import torch
from torchvision.utils import save_image

if __name__ == "__main__":
    # settings
    lmdb_path = "data/ali_vqvae_hard_biggerset_011.lmdb"
    output_path = f"test_lmdb_{lmdb_path.split('/')[-1]}.jpg"
    args = argparse.Namespace()
    args.img_tokenizer_path = 'pretrained/vqvae/vqvae_hard_biggerset_011.pt'
    args.img_tokenizer_num_tokens = None
    device = 'cuda:0'

    torch.cuda.set_device(device)
    tokenizer = get_tokenizer(args)
    with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
        imgs = []
        txts = []
        for i in range(20,50):
            txt, images = tokenizer.DecodeIds(search(env, i))
            txts.append(txt)
            imgs.append(images[0])
        print(txts)
        imgs = torch.cat(imgs, dim=0)
        save_image(imgs, output_path,  normalize=True, range=None)