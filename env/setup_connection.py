# -*- encoding: utf-8 -*-
'''
@File    :   setup_connection.py
@Time    :   2021/01/16 16:50:36
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import base64

if __name__ == "__main__":
    ssh_config = ''
    line_format = 'Host node{}\n\tUser root\n\tPort 2222\n\tHostname {}\n'
    for i, ip in enumerate(sys.argv[1:]):
        ssh_config += line_format.format(i, ip)
    
    ret = os.system(f'echo \"{ssh_config}\" > ~/.ssh/config && chmod 600 ~/.ssh/config')
    assert ret == 0

    hostfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hostfile')
    with open(hostfile_path, 'w') as fout:
        for i, ip in enumerate(sys.argv[1:]):
            fout.write(f'node{i} slots=8\n')
    print(f'Successfully generating hostfile \'{hostfile_path}\'!')


