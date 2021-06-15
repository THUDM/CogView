# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2021/01/11 16:35:24
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib

from .unified_tokenizer import get_tokenizer

from .templates import *
from .configure_data import make_loaders, detect_new_datasets