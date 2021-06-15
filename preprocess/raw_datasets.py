# -*- encoding: utf-8 -*-
'''
@File    :   raw_datasets.py
@Time    :   2021/01/24 15:31:34
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
from tqdm import tqdm
import ctypes
import io

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, IterableDataset
from torchvision import datasets
import unrar
from PIL import Image
import timeit
from collections import Iterable


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        filename = filename.split('.')[0]
        return sample, filename

class RarDataset(Dataset):
    def __init__(self, path, transform=None):
        from unrar import rarfile
        self.rar = rarfile.RarFile(path)
        self.infos = self.rar.infolist()
        self.transform = transform
    def __len__(self):
        return len(self.infos)
    def __getitem__(self, idx):
        target_info = self.infos[idx]
        img = Image.open(self.rar.open(target_info))
        dirs, filename = os.path.split(self.infos[idx].filename)
        filename = filename.split('.')[0]
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

from unrar import rarfile
from unrar import unrarlib
from unrar import constants
from unrar.rarfile import _ReadIntoMemory, BadRarFile
import zipfile
import PIL

class ZipDataset(Dataset):
    def __init__(self, path, transform=None):
        self.zip = zipfile.ZipFile(path)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.members = [info for info in self.zip.infolist() if info.filename[-1] != os.sep]
        else:
            all_members = [info for info in self.zip.infolist() if info.filename[-1] != os.sep]
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            self.members = [x for i, x in enumerate(all_members) if i % num_workers == worker_id]

        self.transform = transform
    def __len__(self):
        return len(self.members)
    def __getitem__(self, idx):
        target_info = self.members[idx]
        img = Image.open(self.zip.open(target_info))
        dirs, filename = os.path.split(self.members[idx].filename)
        filename = filename.split('.')[0]
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

import h5py

class H5Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.h5 = h5py.File(path, "r")
        self.images = self.h5["input_image"]
        self.members = None
        self.transform = transform

    def create_members(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.members = self.h5['index'][:]
        else:
            all_members = self.h5['index'][:]
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            self.members = [x for i, x in enumerate(all_members) if i % num_workers == worker_id]

    def __len__(self):
        if self.members is None:
            self.create_members()
        return len(self.members)

    def __getitem__(self, idx):
        if self.members is None:
            self.create_members()
        target_info = self.members[idx]
        try:
            img = Image.fromarray(self.images[target_info][0])
            if self.transform is not None:
                img = self.transform(img)
            return img, int(target_info)
        except(OSError, IndexError):
            print("warning: OSError or IndexError")
            return Image.new('RGB', (256, 256), (255, 255, 255)), -1

# class StreamingZipDataset(IterableDataset):
#     def __init__(self, path, transform=None):
#         self.zip = zipfile.ZipFile(path, "r")
#         self.transform = transform
#     def __len__(self):
#         return len(self.zip.filelist)
#     def __next__(self):
#         img = Image.open(self.rar.open(target_info))
#
#         pass
#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info is None:
#             self.members = self.zip.namelist()
#         else:
#             all_members = self.zip.namelist()
#             num_workers = worker_info.num_workers
#             worker_id = worker_info.id
#             self.members = [x for i, x in enumerate(all_members) if i % num_workers == worker_id]
#         self.pointer = 0
#         return self
#     def __del__(self):
#         self.zip.close()

class StreamingRarDataset(IterableDataset):
    def __init__(self, path, transform=None, default_size=256):
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        print("begin open rar")
        self.rar = rarfile.RarFile(path)
        print("finish open rar")
        self.transform = transform
        def callback_fn(file_buffer, filename):
            try:
                img = Image.open(file_buffer.get_bytes()).convert('RGB')
                dirs, filename = os.path.split(filename)
                filename = filename.split('.')[0]
                if self.transform is not None:
                    img = self.transform(img)
                return img, filename
            except PIL.UnidentifiedImageError:
                print("UnidentifiedImageError")
                return torch.zeros((3, default_size, default_size)), "not_a_image"
        self.callback_fn = callback_fn
        # new handle
        self.handle = None
        self.callback_fn = callback_fn

    def __len__(self):
        return len(self.rar.filelist)
    def __next__(self):
        if self.pointer >= len(self.members):
            raise StopIteration()
        if self.handle == None:
            archive = unrarlib.RAROpenArchiveDataEx(
            self.rar.filename, mode=constants.RAR_OM_EXTRACT)
            self.handle = self.rar._open(archive)
        # callback to memory
        self.data_storage = _ReadIntoMemory()
        c_callback = unrarlib.UNRARCALLBACK(self.data_storage._callback)
        unrarlib.RARSetCallback(self.handle, c_callback, 0)
        handle = self.handle
        try:
            rarinfo = self.rar._read_header(handle)
            while rarinfo is not None:
                if rarinfo.filename == self.members[self.pointer]:
                    self.rar._process_current(handle, constants.RAR_TEST)
                    break
                else:
                    self.rar._process_current(handle, constants.RAR_SKIP)
                rarinfo = self.rar._read_header(handle)

            if rarinfo is None:
                self.data_storage = None

        except unrarlib.UnrarException:
            raise BadRarFile("Bad RAR archive data.")

        if self.data_storage is None:
            raise KeyError('There is no item named %r in the archive' % self.members[self.pointer])

        # return file-like object
        ret = self.data_storage
        if self.callback_fn is not None:
            ret = self.callback_fn(ret, self.members[self.pointer])
        self.pointer += 1
        return ret

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.members = self.rar.namelist()
        else:
            all_members = self.rar.namelist()
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            self.members = [x for i, x in enumerate(all_members) if i % num_workers == worker_id]
        self.pointer = 0
        return self

    def __del__(self):
        self.rar._close(self.handle)
