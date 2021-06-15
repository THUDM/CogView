# -*- encoding: utf-8 -*-
'''
@File    :   pretokenized_data.py
@Time    :   2021/01/20 15:39:10
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from vqvae import *
from data_utils import Code2CodeTemplate, concat_codes
from torchvision.transforms.functional import resize
from torchvision import transforms
from data_utils import get_tokenizer

# def make_hierarchical_batch(model, txts, imgs):
#     '''
#         model: VQVAE
#         txts: ['text1', 'text2', ...]
#         imgs: [b, 3, s, s]
#     '''
#     s = img.shape[-1]
#     assert img.shape[-2] == s # square
#     codes_base = img2code(model, img)
#     img_tiny = resize(img, size=s//4).numpy()
#     codes_tiny = img2code(model, img_tiny).numpy()
#     ret = []
#     for i in range(len(txts)):
#         text = '[ROI1] ' + txts[i]
#         ret.append(
#             Code2CodeTemplate(text, codes_tiny[i], codes_base[i])
#         )
#     return ret


def make_super_resolution_batch(model, txts, imgs):
    '''
        [text...small_img...base_img]
    '''
    tokenizer = get_tokenizer()

    if not hasattr(make_super_resolution_batch, 'pos'):
        pos = ['左上', '正上', '右上', '左侧', '中间', '右侧', '左下', '正下', '右下']
        pos = [
            tokenizer.parse_query('[ROI1] 是{}部分图'.format(p))
            for p in pos
        ] # [[23, 354...], [232, ...]]
        pw = [0, 64, 128] * 3
        ph = [0, 0, 0, 64, 64, 64, 128, 128, 128]
        make_super_resolution_batch.pos = list(zip(pos, ph, pw))
        make_super_resolution_batch.weights = [1] * 9
        make_super_resolution_batch.prefix = tokenizer.parse_query('[ROI2] 是 [ROI1] 的放大图')

    s = imgs.shape[-1]
    assert s == imgs.shape[-2] == 256
    # Crop 128 * 128 patch
    selected_poses = random.choices(range(9), weights=make_super_resolution_batch.weights)
    pos = make_super_resolution_batch.pos
    patches = [
        imgs[i, :, pos[p][1]:pos[p][1] + 128, pos[p][2]: pos[p][2]+128]
        for i, p in enumerate(selected_poses)
    ]
    patches = torch.stack(patches)
    small_patches = resize(patches, size=64)

    codes_base = img2code(model, patches).cpu().numpy()
    codes_small = img2code(model, small_patches).cpu().numpy()

    ret = []
    for i in range(len(txts)):
        code_text = tokenizer(txts[i])
        ret.append(
            concat_codes(code_text + make_super_resolution_batch.prefix,
                codes_small[i],
                pos[selected_poses[i]][0],
                codes_base[i])
        )
    return ret

def make_super_resolution_batch(model, txts, imgs, img_size=512, sampling_num=4):
    '''
        [text...small_img...base_img]
    '''
    tokenizer = get_tokenizer()
    t0, t1 = img_size // 4, img_size // 2
    if img_size == 512:
        size_tk = tokenizer['[BASE]']
    else:
        raise NotImplementedError

    pw = [0, t0, t1] * 3
    ph = [0, 0, 0, t0, t0, t0, t1, t1, t1]
    ptk = [[tokenizer['[EOI1]'], tokenizer['[ROI2]'], tokenizer[f'[POS{i}]'], size_tk, tokenizer['[BOI2]']]
     for i in range(9)
     ]
    pos = list(zip(ptk, ph, pw))
    weights = [1] * 9
        

    s = imgs.shape[-1]
    assert s == imgs.shape[-2] == img_size
    # Crop img_size/2 * img_size/2 patch
    selected_poses = random.choices(range(9), weights=weights, k=sampling_num)
    pos = pos
    patches = [
        imgs[i, :, pos[p][1]:pos[p][1] + t1, pos[p][2]: pos[p][2]+t1]
        for i in range(imgs.shape[0])
            for p in selected_poses
    ]
    patch_prefix = [
        pos[p][0]
        for p in selected_poses
    ] * imgs.shape[0]
    patches = torch.stack(patches)
    overviews = torch.nn.functional.interpolate(imgs, size=(t1, t1), mode='bilinear')

    codes_patches = img2code(model, patches).cpu().numpy()
    codes_overviews = img2code(model, overviews).cpu().numpy()
    ret = []
    for i in range(len(txts)):
        code_text = [tokenizer['[ROI1]']] + tokenizer(txts[i]) + [size_tk, tokenizer['[BOI1]']]
        for j in range(sampling_num):
            ret.append(
                concat_codes(code_text,
                    codes_overviews[i],
                    patch_prefix[i* sampling_num + j],
                    codes_patches[i * sampling_num + j],
                    [tokenizer['[EOI2]']]
                    )
            )
    return ret

def make_text_image_batch(model, txts, imgs):
    from data_utils import TextCodeTemplate
    s = imgs.shape[-1]
    assert s == imgs.shape[-2] == 256 
    tokenizer = get_tokenizer()
    codes = img2code(model, imgs).cpu().numpy()
    ret = []
    for i in range(len(txts)):
        ret.append( 
            TextCodeTemplate(txts[i], codes[i])
        )
    return ret

def make_tuple_text_image_batch(model, txts, imgs):
    s = imgs.shape[-1]
    assert s == imgs.shape[-2] == 256
    codes = img2code(model, imgs).cpu().numpy()
    ret = []
    for i in range(len(txts)):
        ret.append(
            (txts[i], codes[i])
        )
    return codes

import itertools
def make_cut_text_batch(txts, seq_len):
    from data_utils import PureTextTemplate
    tmp_list = np.array(list(
        itertools.chain(*(PureTextTemplate(txt) for txt in txts))
        ))
    ret = [
        tmp_list[en - seq_len: en]
        for en in range(seq_len, len(tmp_list), seq_len)
    ]
    return ret

