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

"""argparser configuration"""

import argparse
import os
import torch
import deepspeed
import json


def add_model_config_args(parser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')

    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='dropout probability for attention weights')
    group.add_argument('--num-attention-heads', type=int, default=16,
                       help='num of transformer attention heads')
    group.add_argument('--hidden-size', type=int, default=1024,
                       help='tansformer hidden size')
    group.add_argument('--num-layers', type=int, default=24,
                       help='num decoder layers')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='layer norm epsilon')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='dropout probability for hidden state transformer')
    group.add_argument('--max-position-embeddings', type=int, default=512,
                       help='maximum number of position embeddings to use')
    group.add_argument('--vocab-size', type=int, default=30522,
                       help='vocab size to use for non-character-level '
                            'tokenization. This value will only be used when '
                            'creating a tokenizer')
    group.add_argument('--deep-init', action='store_true',
                       help='initialize bert model similar to gpt2 model.'
                            'scales initialization of projection layers by a '
                            'factor of 1/sqrt(2N). Necessary to train bert '
                            'models larger than BERT-Large.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficieny reasons.')
    group.add_argument('--cpu-optimizer', action='store_true',
                       help='Run optimizer on CPU')
    group.add_argument('--cpu_torch_adam', action='store_true',
                       help='Use Torch Adam as optimizer on CPU.')
    
    group.add_argument('--max-position-embeddings-finetune', type=int, default=-1,
                       help='maximum number of position embeddings to use in finetune')

    return parser


def add_fp16_config_args(parser):
    """Mixed precision arguments."""

    group = parser.add_argument_group('fp16', 'fp16 configurations')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode')
    group.add_argument('--fp32-embedding', action='store_true',
                       help='embedding in fp32')
    group.add_argument('--fp32-layernorm', action='store_true',
                       help='layer norm in fp32')
    group.add_argument('--fp32-tokentypes', action='store_true',
                       help='embedding token types in fp32')
    group.add_argument('--fp32-allreduce', action='store_true',
                       help='all-reduce in fp32')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                            'values can improve fp16 convergence. If None, dynamic'
                            'loss scaling is used.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale')
    group.add_argument('--min-scale', type=float, default=1,
                       help='Minimum loss scale for dynamic loss scale')

    return parser


def add_training_args(parser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    group.add_argument('--experiment-name', type=str, default="CogView",
                       help="The experiment name for summary and checkpoint")
    group.add_argument('--batch-size', type=int, default=4,
                       help='Data Loader batch size')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='weight decay coefficient for L2 regularization')
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='checkpoint activation to allow for training '
                            'with larger models and sequences')
    group.add_argument('--checkpoint-num-layers', type=int, default=1,
                       help='chunk size (number of layers) for checkpointing')
    group.add_argument('--deepspeed-activation-checkpointing', action='store_true',
                       help='uses activation checkpointing from deepspeed')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--train-iters', type=int, default=1000000,
                       help='total number of iterations to train over all training runs')
    group.add_argument('--log-interval', type=int, default=50,
                       help='report interval')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after this many new iterations.')
    group.add_argument('--summary-dir', type=str, default="", help="The directory to store the summary")
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed')
    group.add_argument('--img-tokenizer-path', type=str, default=None,
                       help='The checkpoint file path of image tokenizer.')
    group.add_argument('--img-tokenizer-num-tokens', type=int, default=None,
                       help='The num tokens of image tokenizer. ONLY use for pretraining with img-tokenizer UNKNOW.')
    # Batch prodecuer arguments
    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention maske after '
                            'end-of-document token.')

    # Learning rate.
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                            ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='learning rate decay function')
    group.add_argument('--lr-decay-ratio', type=float, default=0.1)
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                            'training iters). Default 0.01')
    # model checkpointing
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=5000,
                       help='number of iterations between saves')
    group.add_argument('--no-save-optim', action='store_true',
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true',
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true',
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true',
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                            'or rng state from checkpoint and set iteration to 0. '
                            'Assumed when loading a release checkpoint.')
    group.add_argument('--resume-dataloader', action='store_true',
                       help='Resume the dataloader when resuming training. '
                            'Does not apply to tfrecords dataloader, try resuming'
                            'with a different seed in this case.')
    # distributed training args
    group.add_argument('--distributed-backend', default='nccl',
                       help='which backend to use for distributed '
                            'training. One of [gloo, nccl]')

    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    
    # loss scale
    group.add_argument('--txt-loss-scale', type=float, default=1)
    group.add_argument('--fast-load', action='store_true',
                       help='load checkpoints without locks.')

    return parser


def add_evaluation_args(parser):
    """Evaluation arguments."""

    group = parser.add_argument_group('validation', 'validation configurations')

    group.add_argument('--eval-batch-size', type=int, default=None,
                       help='Data Loader batch size for evaluation datasets.'
                            'Defaults to `--batch-size`')
    group.add_argument('--eval-iters', type=int, default=100,
                       help='number of iterations to run for evaluation'
                            'validation/test for')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='interval between running evaluation on validation set')

    return parser


def add_text_generate_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('Text generation', 'configurations')
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--top_p", type=float, default=0.0)
    group.add_argument("--top_k", type=int, default=0)
    # group.add_argument("--out-seq-length", type=int, default=256)
    group.add_argument("--generation-task", type=str,
                       default='text2image',
                       choices=['text2image',
                                'image2text',
                                'super-resolution',
                                'low-level super-resolution',
                                'post-selection',
                                'raw'
                                ],
                       help='what type of inference task to use')
    group.add_argument('--input-source', type=str, default='interactive',
                       help='what input mode to use, interactive or path')
    group.add_argument('--output-path', type=str, default='./samples',
                       help='path to place the generated samples')
    group.add_argument('--debug', action='store_true',
                       help='Debug will merge all outputs.')
    group.add_argument('--with-id', action='store_true',
                       help='If each line is prepended with an id.')
    group.add_argument('--max-inference-batch-size', type=int, default=12)
    return parser


def add_data_args(parser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group('data', 'data configurations')

    group.add_argument('--model-parallel-size', type=int, default=1,
                       help='size of the model parallel.')
    group.add_argument('--shuffle', action='store_true',
                       help='Shuffle data. Shuffling is deterministic '
                            'based on seed and current epoch.')
    group.add_argument('--train-data', nargs='+', default=None,
                       help='Whitespace separated filenames or corpora names '
                            'for training.')

    group.add_argument('--valid-data', nargs='*', default=None,
                       help="""Filename for validation data.""")
    group.add_argument('--split', default='1000,1,1',
                       help='comma-separated list of proportions for training,'
                            ' validation, and test split')
    group.add_argument('--test-data', nargs='*', default=None,
                       help="""Filename for testing""")

    group.add_argument('--num-workers', type=int, default=2,
                       help="""Number of workers to use for dataloading""")

    group.add_argument('--dataset-type', type=str,
                       default='TokenizedDataset',
                       choices=['TokenizedDataset',
                                'TextCodeDataset',
                                'CompactBinaryDataset'
                                ],
                       help='what type of dataset to use')

    group.add_argument('--max-memory-length', type=int, default=2048,
                       help="max memory buffer for attention")
    group.add_argument('--new-dataset-path', type=str, default=None,
                       help='The folder we will dynamically check for lmdbs during training.')
    
    return parser

def add_generation_api_args(parser):
    """generation api arguments"""

    group = parser.add_argument_group('api', 'api configurations')

    group.add_argument('--img_folder_path', default='image/')
    group.add_argument('--input_folder_path', default='input/')
    group.add_argument('--input_rec_path', default='input/')
    group.add_argument('--check_mode', default='code')
    group.add_argument('--time_interval', default=10)
    group.add_argument('--device', default=None)

    return parser

def add_sparse_args(parser):
    """sparse attention arguments."""

    group = parser.add_argument_group('Sparse Attention', 'sparse configurations')
    group.add_argument('--is-sparse', type=int, default=0,
                       choices=[0, 1, 2],
                       help='whether use sparse attention. 0 not 1 train 2 inference') # TODO: Temporally not using is-sparse==2 (not optimized), use 0 for inference.
    group.add_argument("--query-window", type=int, default=128)
    group.add_argument("--key-window-times", type=int, default=6)
    group.add_argument("--num-pivot", type=int, default=768)
    return parser

def get_args():
    """Parse all the args."""

    parser = argparse.ArgumentParser(description='PyTorch CogView Model')
    parser = add_model_config_args(parser)
    parser = add_fp16_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_text_generate_args(parser)
    parser = add_data_args(parser)
    parser = add_generation_api_args(parser)
    parser = add_sparse_args(parser)

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    if not args.train_data:
        print('WARNING: No training data specified')
        assert args.is_sparse != 1, 'use is-sparse == 2 for inference'
    elif args.is_sparse == 1 and (args.max_position_embeddings - 1) % args.query_window != 0:
        raise ValueError('During sparse training, the sequence length must be exactly divided by window_size.') 

    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    if hasattr(args, 'deepspeed_mpi') and args.deepspeed_mpi:
        mpi_define_env(args)
    elif os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'):
        # We are using (OpenMPI) mpirun for launching distributed data parallel processes
        local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        local_size = int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'))

        # Possibly running with Slurm
        num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES', '1'))
        nodeid = int(os.getenv('SLURM_NODEID', '0'))

        args.local_rank = local_rank
        args.rank = nodeid * local_size + local_rank
        args.world_size = num_nodes * local_size

    args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print('using world size: {} and model-parallel size: {} '.format(
            args.world_size, args.model_parallel_size))

    args.dynamic_loss_scale = False
    if args.loss_scale is None:
        args.dynamic_loss_scale = True
        if args.rank == 0:
            print(' > using dynamic loss scaling')

    # The args fp32_* or fp16_* meant to be active when the
    # args fp16 is set. So the default behaviour should all
    # be false.
    if not args.fp16:
        args.fp32_embedding = False
        args.fp32_tokentypes = False
        args.fp32_layernorm = False

    if hasattr(args, "deepspeed") and args.deepspeed and args.deepspeed_config is not None:
        with open(args.deepspeed_config) as file:
            deepspeed_config = json.load(file)
        if "train_micro_batch_size_per_gpu" in deepspeed_config:
            args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
        if "gradient_accumulation_steps" in deepspeed_config:
            args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
        else:
            args.gradient_accumulation_steps = None
        if "optimizer" in deepspeed_config:
            optimizer_params_config = deepspeed_config["optimizer"].get("params", {})
            args.lr = optimizer_params_config.get("lr", args.lr)
            args.weight_decay = optimizer_params_config.get("weight_decay", args.weight_decay)
    return args


def mpi_define_env(args):
    ''' For training CogView via MPI to setup the connection.
        Omit this function if use the basic deepspeed pdsh runner. 
    '''
    from mpi4py import MPI
    import subprocess
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    master_addr = None
    if rank == 0:
        hostname_cmd = ["hostname -I"]
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_addr = result.decode('utf-8').split()[0]
    master_addr = comm.bcast(master_addr, root=0)

    # Determine local rank by assuming hostnames are unique
    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([i == proc_name for i in all_procs[:rank]])

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    args.local_rank = local_rank
    args.world_size = world_size
    args.rank = rank
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = "29500" # TORCH_DISTRIBUTED_DEFAULT_PORT = 29500

    print(
        "Discovered MPI settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
        .format(os.environ['RANK'],
                args.local_rank,
                os.environ['WORLD_SIZE'],
                os.environ['MASTER_ADDR'],
                os.environ['MASTER_PORT']))
