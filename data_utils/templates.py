# -*- encoding: utf-8 -*-
'''
@File    :   templates.py
@Time    :   2021/01/11 22:28:57
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

from .unified_tokenizer import get_tokenizer
from .vqvae_tokenizer import sqrt_int

def concat_codes(*codes):
    is_numpy = is_tensor = False
    for code in codes:
        if isinstance(code, np.ndarray):
            is_numpy = True
        if isinstance(code, torch.Tensor):
            is_tensor = True
            device = code.device
    if is_tensor:
        return torch.cat(
            [
                torch.tensor(code, device=device)
                for code in codes
            ]
        )
    elif is_numpy:
        return np.concatenate(
            [
                np.array(code)
                for code in codes
            ],
            axis=0
        )
    else:
        ret = []
        for code in codes:
            ret = ret + code
        return ret

def TextCodeTemplate(text, code):
    tokenizer = get_tokenizer()
    if isinstance(text, str):
        text_ids = [tokenizer['[ROI1]']] + tokenizer(text)
    else:
        text_ids = np.concatenate(
                (
                    np.array([tokenizer['[ROI1]']]),
                    text,
                ),
                axis=0
            )
    code = tokenizer.wrap_code(code)
    return concat_codes(text_ids, code)

def Code2CodeTemplate(text, code0, code1):
    tokenizer = get_tokenizer()
    text_ids = tokenizer.parse_query(text) if isinstance(text, str) else text
    code0 = tokenizer.wrap_code(code0)
    code1 = tokenizer.wrap_code(code1, idx=2)
    return concat_codes(text_ids, code0, code1)

def PureTextTemplate(text):
    tokenizer = get_tokenizer()
    return tokenizer(text) + [tokenizer['[SEP]']]







