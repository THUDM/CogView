# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""

import os
import stat
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from datetime import datetime
from arguments import get_args
from utils import Timers
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from utils import load_checkpoint, get_checkpoint_iteration
from data_utils import get_tokenizer
import mpu
import deepspeed

from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0
from pretrain_gpt2 import get_model
import math
from copy import deepcopy
from tqdm import tqdm
from generation import get_batch, filling_sequence, add_interlacing_beam_marks, magnify, inverse_prompt_score
from torchvision.utils import save_image
import torch.distributed as dist


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)

    if args.load is not None:
        if args.deepspeed:
            iteration, release, success = get_checkpoint_iteration(args)
            path = os.path.join(args.load, str(iteration), "mp_rank_00_model_states.pt")
            print('current device:', torch.cuda.current_device())

            if not args.offload:
                checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            else:
                checkpoint = torch.load(path, map_location=torch.device('cpu'))

            model.load_state_dict(checkpoint["module"])

            if args.offload:
                model = model.half().cuda()
            model.load_state_dict(checkpoint["module"])
            print(f"Load model file {path}")
        else:
            _ = load_checkpoint(
                model, None, None, args, load_optimizer_states=False)

    return model

def _parse_and_to_tensor(text, img_size=256, query_template='{}'):
    tokenizer = get_tokenizer()
    text = query_template.format(*text.split('\t'))
    seq = tokenizer.parse_query(text, img_size=img_size)
    seq = torch.cuda.LongTensor(seq)
    return seq

def get_context(args, query_template='{}'):
    tokenizer = get_tokenizer()
    terminate_runs = 0
    img_size = 256 if args.generation_task != 'low-level super-resolution' else 128
    ml = max(args.max_position_embeddings, args.max_position_embeddings_finetune)
    output_path = args.output_path

    if args.input_source == 'interactive':
        assert not args.with_id, '--with-id is only used with file inputs.'
        if args.generation_task == 'post-selection':
            raise ValueError('post-selection only takes file inputs!')
        while True:
            raw_text = input("\nPlease Input Query (stop to exit) >>> ") 
            if not raw_text:
                print('Query should not be empty!')
                continue
            if raw_text == "stop":
                return 
            try:
                seq = _parse_and_to_tensor(raw_text, img_size=img_size, query_template=query_template)
            except (ValueError, FileNotFoundError) as e:
                print(e)
                continue
            if len(seq) > ml:
                print("\nSeq length", len(seq),
                      f"\nPlease give smaller context than {ml}!")
                continue
            yield (raw_text, seq, output_path)
    else:
        with open(args.input_source, 'r') as fin:
            inputs = fin.readlines()
        for line_no, raw_text in enumerate(inputs):
            if line_no % dist.get_world_size() != dist.get_rank():
                continue
            rk = dist.get_rank()
            print(f'Working on No. {line_no} on {rk}... ')
            raw_text = raw_text.strip()
            if len(raw_text) == 0:
                continue
            if args.with_id: # with id
                parts = raw_text.split('\t')
                output_path = os.path.join(args.output_path, parts[0])
                raw_text = '\t'.join(parts[1:])

            if args.generation_task == 'post-selection':
                parts = raw_text.split('\t')
                seqs = []
                for part in parts[1:]:
                    try:
                        seq_single = _parse_and_to_tensor('\t'.join([part, parts[0]]), img_size=img_size, query_template=query_template)
                        seqs.append(seq_single)
                    except (ValueError, FileNotFoundError) as e:
                        print(e)
                        continue
                seq = torch.stack(seqs)
            else:
                try:
                    seq = _parse_and_to_tensor(raw_text, img_size=img_size, query_template=query_template)
                except (ValueError, FileNotFoundError) as e:
                    print(e)
                    continue
                if len(seq) > ml:
                    print("\nSeq length", len(seq),
                        f"\nPlease give smaller context than {ml}!")
                    continue
            yield (raw_text, seq, output_path)


def generate_images_once(model, args, raw_text, seq=None, num=8, query_template='{}', output_path='./samples'):
    tokenizer = get_tokenizer()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if seq is None: # need parse
        img_size = 256 if args.generation_task != 'low-level super-resolution' else 128
        seq = _parse_and_to_tensor(raw_text, img_size=img_size, query_template=query_template)
    model.eval()
    with torch.no_grad():
        print('show raw text:', raw_text)
        start_time = time.time()
        if args.generation_task in ['text2image', 'low-level super-resolution']:
            invalid_slices = [slice(tokenizer.img_tokenizer.num_tokens, None)]
        elif args.generation_task == 'image2text':
            invalid_slices = [slice(0, tokenizer.img_tokenizer.num_tokens)]
        else:
            NotImplementedError

        mbz = args.max_inference_batch_size
        add_interlacing_beam_marks(seq, nb=min(num, mbz))
        assert num < mbz or num % mbz == 0
        output_tokens_list = []
        for tim in range(max(num // mbz, 1)):
            output_tokens_list.append(filling_sequence(model, seq.clone(), args))
            torch.cuda.empty_cache()

        output_tokens_list = torch.cat(output_tokens_list, dim=0)
        
        print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
        print("\nContext:", raw_text, flush=True)
        imgs, txts = [], []
        for seq in output_tokens_list:
            decoded_txts, decoded_imgs = tokenizer.DecodeIds(seq.tolist())
            for i in range(len(decoded_imgs)):
                if decoded_imgs[i].shape[-1] == 128:
                    decoded_imgs[i] = torch.nn.functional.interpolate(decoded_imgs[i], size=(256, 256))
            if args.debug:
                imgs.extend(decoded_imgs)
            else:
                imgs.append(decoded_imgs[-1]) # only the last image (target)
            txts.append(decoded_txts)
        if args.generation_task == 'image2text':
            print(txts)
            return 
        if args.debug:
            output_file_prefix = raw_text.replace('/', '')[:20]
            output_file = os.path.join(output_path, f"{output_file_prefix}-{datetime.now().strftime('%m-%d-%H-%M-%S')}.jpg")
            imgs = torch.cat(imgs, dim=0)
            print(txts)
            print("\nSave to: ", output_file, flush=True)
            save_image(imgs, output_file, normalize=True)
        else:
            print("\nSave to: ", output_path, flush=True)
            for i in range(len(imgs)):
                save_image(imgs[i], os.path.join(output_path,f'{i}.jpg'), normalize=True)
                os.chmod(os.path.join(output_path,f'{i}.jpg'), stat.S_IRWXO+stat.S_IRWXG+stat.S_IRWXU)
            save_image(torch.cat(imgs, dim=0), os.path.join(output_path,f'concat.jpg'), normalize=True)
            os.chmod(os.path.join(output_path,f'concat.jpg'), stat.S_IRWXO+stat.S_IRWXG+stat.S_IRWXU)

def generate_images_continually(model, args):
    if args.generation_task == 'text2image':
        query_template = '[ROI1] {} [BASE] [BOI1] [MASK]*1024'
    elif args.generation_task == 'image2text':
        query_template = '[BASE] [BOI1] [Image]{} [EOI1] [ROI1] [MASK]*20'
    elif args.generation_task == 'low-level super-resolution':
        query_template = '[ROI1] {} [BASE] [BOI1] [Image]{} [EOI1] [ROI2] [POS0] [BASE] [BOI2] [MASK]*1024'
    elif args.generation_task == 'super-resolution':
        query_template = '[ROI1] {} [BASE] [BOI1] [Image]{}'
    elif args.generation_task == 'post-selection':
        query_template = '[BASE] [BOI1] [Image]{} [EOI1] [ROI1] {}'
    else:
        raise NotImplementedError
    for raw_text, seq, output_path in get_context(args, query_template):
        if args.generation_task == 'super-resolution':
            super_resolution(model, args, raw_text, seq, output_path=output_path)
        elif args.generation_task == 'post-selection':
            post_selection(model, args, raw_text, seq, output_path=output_path)
        else:
            generate_images_once(model, args, raw_text, seq, num=args.batch_size, output_path=output_path)

def super_resolution(model, args, raw_text, seq, output_path="./samples"):
    tokenizer = get_tokenizer()
    model.eval()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with torch.no_grad():
        start_time = time.time()
        output_tokens_list = magnify(model, tokenizer, seq[-32**2:], seq[:-32**2], args)

        print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
        print("\nContext:", raw_text, flush=True)
        output_file_prefix = raw_text.replace('/', '')[:20]
        output_file = os.path.join(output_path, f"{output_file_prefix}-{datetime.now().strftime('%m-%d-%H-%M-%S')}.jpg")
        imgs = []
        if args.debug:
            imgs.append(torch.nn.functional.interpolate(tokenizer.img_tokenizer.DecodeIds(seq[-32**2:]), size=(512, 512)))
        for seq in output_tokens_list:
            decoded_txts, decoded_imgs = tokenizer.DecodeIds(seq.tolist())
            imgs.extend(decoded_imgs)
        imgs = torch.cat(imgs, dim=0)
        print("\nSave to: ", output_file, flush=True)
        save_image(imgs, output_file, normalize=True)

def post_selection(model, args, raw_text, seq, output_path):
    tokenizer = get_tokenizer()
    model.eval()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with torch.no_grad():
        start_time = time.time()

        num = seq.shape[0]
        mbz = args.max_inference_batch_size
        assert num < mbz or num % mbz == 0
        scores = [inverse_prompt_score(model, seq[tim*mbz:(tim+1)*mbz], args)
            for tim in range(max(num // mbz, 1))
            ]
        scores = torch.cat(scores, dim=0)
        # scores = inverse_prompt_score(model, seq, args) # once

        print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
        print("\nContext:", raw_text, flush=True)
        rank = dist.get_rank()
        output_file = os.path.join(output_path, f"scores_rank_{rank}.txt")
        with open(output_file, 'a') as fout:
            fout.write(raw_text+'\n')
            fout.write('\t'.join([str(x) for x in scores.tolist()])+'\n')
        print("\nSave to: ", output_file, flush=True)
       

                
def prepare_tokenizer(args):

    tokenizer = get_tokenizer(args)

    num_tokens = tokenizer.num_tokens
    before = num_tokens
    after = before
    multiple = args.make_vocab_size_divisible_by * \
               mpu.get_model_parallel_world_size()
    while (after % multiple) != 0:
        after += 1
    print_rank_0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(
        before, after - before, after))

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer


def main():
    """Main training program."""

    print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # set device, this args.device is only used in inference
    if args.device is not None:
        device = int(args.device)
        torch.cuda.set_device(device)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    generate_images_continually(model, args)

if __name__ == "__main__":
    main()
