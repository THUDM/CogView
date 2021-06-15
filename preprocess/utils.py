# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/01/24 16:35:43
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
from vqvae import code2img, img2code
from torchvision.utils import save_image


def show_recover_results(model, imgs):
    codes = img2code(model, imgs)
    recovered = code2img(model, codes)
    mean = torch.tensor([0.79093, 0.76271, 0.75340], device=recovered.device).view(-1, 1, 1)
    std = torch.tensor([0.30379, 0.32279, 0.32800], device=recovered.device).view(-1, 1, 1)
    recovered = (recovered * std + mean).clamp(0, 1)
    imgs = (imgs * std + mean).clamp(0, 1)
    out = torch.cat([imgs, recovered], dim=0)
    save_image(out, 'samples/show_recover_results.jpg', normalize=False, nrow=len(imgs))
