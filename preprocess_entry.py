import os
import sys
import math
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess args")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--img_tokenizer_path", type=str, default='vqvae_hard_biggerset_011.pt')
    parser.add_argument("--encode_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    print(args)
    img_size = args.encode_size * 8

    # args = argparse.Namespace()
    # args.img_tokenizer_path = 'pretrained/vqvae/vqvae_hard_018.pt'#old path
    # args.img_tokenizer_path = 'pretrained/vqvae/vqvae_hard_biggerset_011.pt'
    # args.img_tokenizer_path = '/root/mnt/vqvae_1epoch_64x64.pt'
    args.img_tokenizer_num_tokens = None

    device = f'cuda:{args.device}'
    torch.cuda.set_device(device)
    name = args.dataset + "_" + args.img_tokenizer_path.split(".")[0] + ".lmdb"
    args.img_tokenizer_path = f"pretrained/vqvae/{args.img_tokenizer_path}"

    datasets = {}
    datasets["ali"] = [
        ['/root/mnt/sq_gouhou_white_pict_title_word_256_fulltitle.tsv'],
        ['/root/mnt/dingming/ali_white_picts_256.zip'],
        "tsv"
    ]
    datasets["ks3"] = [
        ['/root/mnt/KS3/a_baidu_image_msg_data.json'],
        ['/root/mnt/KS3/downloadImages.rar'],
        "json_ks"
    ]
    datasets["zijian"] = [
        ['/root/mnt/zijian/zj_duomotai_clean_done_data_new.json',
         '/root/mnt/zijian/zj_duomotai_local_server_last_surplus_120w.json'],
        ['/root/mnt/imageFolder_part01.rar',
         '/root/mnt/zijian/imagesFolder_last_surplus_120w.rar'],
        "json"
    ]
    datasets["google"] = [
        ['/root/mnt/google/google_image_message_data.json'],
        ['/root/mnt/google/downloadImage_2020_12_16.rar'],
        "json_ks"
    ]
    datasets["zijian1"] = [
        ['/root/mnt/zijian/zj_duomotai_clean_done_data_new.json'],
        ['/root/cogview2/data/imageFolder_part01.rar'],
        "json"
    ]
    datasets["zijian2"] = [
        ['/root/mnt/zijian/zj_duomotai_local_server_last_surplus_120w.json'],
        ['/root/mnt/zijian/imagesFolder_last_surplus_120w.rar'],
        "json"
    ]
    txt_files, img_folders, txt_type = datasets[args.dataset]

    os.environ['UNRAR_LIB_PATH'] = '/usr/local/lib/libunrar.so'


    from data_utils import get_tokenizer
    tokenizer = get_tokenizer(args)
    model = tokenizer.img_tokenizer.model

    print("finish init vqvae_model")

    from preprocess.preprocess_text_image_data import extract_code,extract_code_super_resolution_patches

    # =====================   Define Imgs   ======================== #
    from preprocess.raw_datasets import H5Dataset, StreamingRarDataset, ZipDataset

    datasets = []
    for img_folder in img_folders:
        if img_folder[-3:] == "rar":
            dataset = StreamingRarDataset(path=img_folder, transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.79093, 0.76271, 0.75340], [0.30379, 0.32279, 0.32800]),
            ]), 
            default_size=img_size)
        elif img_folder[-3:] == "zip":
            dataset = ZipDataset(path=img_folder, transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.79093, 0.76271, 0.75340], [0.30379, 0.32279, 0.32800]),
            ]))
        else:
            dataset = H5Dataset(path=img_folder, transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.79093, 0.76271, 0.75340], [0.30379, 0.32279, 0.32800]),
            ]))
        datasets.append(dataset)
    print('Finish reading meta-data of dataset.')
    # ===================== END OF BLOCK ======================= #

    # from preprocess import show_recover_results


    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    # loader = iter(loader)
    # samples = []
    # for k in range(8):
    #     x = next(loader)
    #     print(x[1])
    #     x = x[0].to(device)
    #     samples.append(x)
    # samples = torch.cat(samples, dim=0)
    # show_recover_results(model, samples)

    # =====================   Load Text   ======================== #
    if txt_type == "json":
        import json
        txt_list = []
        for txt in txt_files:
            with open(txt, 'r') as fin:
                t = json.load(fin)
                txt_list.extend(list(t.items()))
        tmp = []
        for k, v in tqdm(txt_list):
            tmp.append((v['uniqueKey'], v['cnShortText']))
        text_dict = dict(tmp)
    elif txt_type == "json_ks":
        import json
        txt_list = []
        for txt in txt_files:
            with open(txt, 'r') as fin:
                t = json.load(fin)
            txt_list.extend(t["RECORDS"])
        tmp = []
        for v in tqdm(txt_list):
            tmp.append((v['uniqueKey'], v['cnShortText']))
        text_dict = dict(tmp)
    elif txt_type == "tsv":
        import pandas as pd
        txt_list = []
        for txt in txt_files:
            t = pd.read_csv(txt, sep='\t')
            txt_list.extend(list(t.values))
        tmp = []
        for k, v in tqdm(txt_list):
            tmp.append((str(k), v))
        text_dict = dict(tmp)
    else:
        des = dataset.h5["input_concat_description"]
        txt_name = dataset.h5["input_name"]
        tmp = []
        for i in tqdm(range(len(des))):
            tmp.append((i, des[i][0].decode("latin-1")+txt_name[i][0].decode("latin-1")))
        text_dict = dict(tmp)
    print('Finish reading texts of dataset.')
    # ===================== END OF BLOCK ======================= #

    # extract_code(model, datasets, text_dict, name, device, txt_type)
    extract_code_super_resolution_patches(model, datasets, text_dict, name, device, txt_type)