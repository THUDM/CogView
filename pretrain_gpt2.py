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

"""Pretrain GPT2"""

# Flag to use Pytorch ddp which uses overlapping communication and computation.
USE_TORCH_DDP = True

from datetime import datetime
import os
import random
import math
from filelock import FileLock
import numpy as np
import torch

import deepspeed
from contextlib import ExitStack
from arguments import get_args
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import GPT2Model
from model import gpt2_get_params_for_weight_decay_optimization

if USE_TORCH_DDP:
    from model import PyTorchDistributedDataParallel as DDP
else:
    from model import DistributedDataParallel as DDP
import mpu
from apex.optimizers import FusedAdam as Adam
from utils import Timers
from utils import save_checkpoint
from utils import load_checkpoint
from utils import report_memory
from utils import print_args
from utils import print_params_min_max_norm
from utils import print_rank_0
from utils import get_sample_writer
import torch.distributed as dist

from data_utils import make_loaders, get_tokenizer, detect_new_datasets

import stat

def get_model(args):
    """Build the model."""

    print_rank_0('building CogView2 model ...')
    # print(args.vocab_size)
    # ml = max(args.max_position_embeddings, args.max_position_embeddings_finetune)
    ml = args.max_position_embeddings
    model = GPT2Model(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      max_sequence_length=ml,
                      max_memory_length=args.max_memory_length,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=True,
                      query_window=args.query_window,
                      key_window_times=args.key_window_times,
                      num_pivot=args.num_pivot
                      )

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if hasattr(args, "deepspeed") and args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if not args.deepspeed:
        if USE_TORCH_DDP:
            i = torch.cuda.current_device()
            model = DDP(model, device_ids=[i], output_device=i,
                        process_group=mpu.get_data_parallel_group())
        else:
            model = DDP(model)

    return model


def get_optimizer_param_groups(model):
    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    param_groups = gpt2_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    return param_groups


def get_optimizer(param_groups, args):
    """Set up the optimizer."""
    if args.cpu_optimizer:
        #Apex FusedAdam uses decoupled weight decay so use the same here
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.AdamW
        else:
            #TODO add option for decoupled weight decay in DeepCPUAdam
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                        lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Use FusedAdam.
        optimizer = Adam(param_groups,
                         lr=args.lr, weight_decay=args.weight_decay)

    print(f'Optimizer = {optimizer.__class__.__name__}')
    if hasattr(args, "deepspeed") and args.deepspeed:
        raise NotImplementedError
        # fp16 wrapper is not required for DeepSpeed.
        # return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               decay_ratio=args.lr_decay_ratio)

    return lr_scheduler


def setup_model_and_optimizer(args):
    """Setup model and optimizer."""

    model = get_model(args)

    param_groups = get_optimizer_param_groups(model)

    if args.train_data is not None:
        if args.deepspeed:
            print_rank_0("DeepSpeed is enabled.")

            model, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=param_groups,
                args=args,
                mpu=mpu,
                dist_init_required=False
            )
        else:
            optimizer = get_optimizer(param_groups, args)
        lr_scheduler = get_learning_rate_scheduler(optimizer, args)
    else:
        optimizer, lr_scheduler = None, None

    return model, optimizer, lr_scheduler


def get_masks_and_position_ids(data,
                            loss_mask=None,
                            attention_mask=None, args=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if attention_mask is None:
        # single direction, [PAD]s are at the end of the seq, so doesn't matter.
        attention_mask = torch.ones((1, seq_length, seq_length), device=data.device)
        attention_mask = torch.tril(attention_mask)
        attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    if args is not None and args.finetune and args.max_position_embeddings < args.max_position_embeddings_finetune:
        # for each sample, find [ROI2] and split
        # ([ROI1] text... [BOI1] img... [EOI1] [ROI2]<pos_id==1089> ...)
        start_token = get_tokenizer()['[ROI2]']
        tmp = torch.nonzero(data == start_token, as_tuple=False)
        start_token_poses = [100000] * batch_size
        for x, y in tmp:
            start_token_poses[x] = min(start_token_poses[x], y)
        assert 100000 not in start_token_poses, 'Some samples do not have [ROI2]!'
        position_ids = torch.zeros(batch_size, seq_length, dtype=torch.long,
                                    device=data.device)
        for i in range(batch_size):
            sep = start_token_poses[i]
            torch.arange(start=0, end=sep, out=position_ids[i, :sep], 
                dtype=torch.long, device=data.device)
            second_pos = 0 # reuse
            torch.arange(start=second_pos, end=second_pos + seq_length - sep, 
                out=position_ids[i, sep:], 
                dtype=torch.long, device=data.device)
        position_ids[position_ids >= args.max_position_embeddings] = args.max_position_embeddings - 1
    else:
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=data.device)
        position_ids = position_ids.unsqueeze(0).expand_as(data)

    return attention_mask, loss_mask, position_ids


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['text', 'loss_mask']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()

    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens_ = data_b['text'].long()
    loss_mask = data_b['loss_mask'].float()
    labels = tokens_[:, 1:].contiguous()
    loss_mask = loss_mask[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    attention_mask = None

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        args=args
        )
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, model, args, timers, mems):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator, args, timers)

    timers('batch generator').stop()

    # split img & txt positions, [PAD] not included # TODO check enough
    tokenizer = get_tokenizer()
    img_txt_sep = tokenizer.img_tokenizer.num_tokens
    img_indices_bool = (tokens.detach() < img_txt_sep)
    txt_indices_bool = (~img_indices_bool) & (loss_mask > 0)
 
    # Forward model.
    logits, *mems = model(tokens, position_ids, attention_mask, txt_indices_bool, img_indices_bool, args.is_sparse, *mems)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(),
                                              labels)
    # scaling loss mask
    loss_mask[txt_indices_bool] *= args.txt_loss_scale
    loss_mask = loss_mask.view(-1)    

    # precalc outlier point, uncomment this if needed
    # if args.iteration > 10000:
    #     outliers = (losses.detach().view(-1) * loss_mask) > 20.
    #     if outliers.sum() > 0:
    #         print(f'Remove {outliers.sum()} outliers.')
    #         loss_mask[outliers] = 1e-4


    losses = losses.view(-1) * loss_mask
    loss = torch.sum(losses) / loss_mask.sum()

    # =====================   Log partial losses   ======================== #
    img_indices_bool = img_indices_bool.view(-1)
    txt_indices_bool = txt_indices_bool.view(-1)
    img_loss = losses[img_indices_bool].detach().sum() / max(img_indices_bool.sum(), 1)
    txt_loss = losses[txt_indices_bool].detach().sum() / max(txt_indices_bool.sum(), 1) / args.txt_loss_scale

    # Reduce losses for logging
    torch.distributed.all_reduce(img_loss.data)
    torch.distributed.all_reduce(txt_loss.data)
    img_loss.data = img_loss.data / args.world_size
    txt_loss.data = txt_loss.data / args.world_size

    # ===================== END OF BLOCK ======================= #
    
    return loss, mems, img_loss, txt_loss


def backward_step(optimizer, model, lm_loss, args, timers):
    """Backward step."""

    # Total loss.
    loss = lm_loss

    # Backward pass.
    if args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    reduced_losses = lm_loss.view(1)

    # Reduce losses for logging
    torch.distributed.all_reduce(reduced_losses.data)
    reduced_losses.data = reduced_losses.data / args.world_size

    if args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('allreduce').reset()
    else:
        if not USE_TORCH_DDP:
            timers('allreduce').start()
            model.allreduce_params(reduce_after=False,
                                   fp32_allreduce=args.fp32_allreduce)
            timers('allreduce').stop()

    lm_loss_reduced = reduced_losses

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)

    return lm_loss_reduced


def see_memory_usage(message, force=False):
    if not force:
        return
    dist.barrier()
    if dist.get_rank() == 0:
        print(message)
        print("Memory Allocated ", torch.cuda.memory_allocated()/(1024*1024*1024), "GigaBytes")
        print("Max Memory Allocated ", torch.cuda.max_memory_allocated()/(1024*1024*1024), "GigaBytes")
        print("Cache Allocated ", torch.cuda.memory_cached()/(1024*1024*1024), "GigaBytes")
        print("Max cache Allocated ", torch.cuda.max_memory_cached()/(1024*1024*1024), "GigaBytes")
        print(" ")

def train_step(data_iterator, model, optimizer, lr_scheduler,
               args, timers, mems):
    """Single training step."""
    while True:
        # Forward model for one step.
        timers('forward').start()
        lm_loss, mems, img_loss, txt_loss = forward_step(data_iterator, model, args, timers, mems)
        timers('forward').stop()

        if (img_loss + txt_loss).isnan().any() or (img_loss + txt_loss).isinf().any():
            print('Skipping backward and optimizer step for nan or inf in forwarding!')
            return (img_loss + txt_loss), 1, mems, img_loss, txt_loss

        # Calculate gradients, reduce across processes, and clip.
        timers('backward').start()
        lm_loss_reduced = backward_step(optimizer, model, lm_loss, args, timers)
        timers('backward').stop()

        # Update parameters.
        skipped_iter, complete = 0, False
        timers('optimizer').start()
        if args.deepspeed:
            if model.is_gradient_accumulation_boundary():
                model.step()
                complete = True
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()
                else:
                    skipped_iter = 1
            else:
                model.step()
        else:
            optimizer.step()
            complete = True
            # Update learning rate.
            if not (args.fp16 and optimizer.overflow):
                lr_scheduler.step()
            else:
                skipped_iter = 1
        timers('optimizer').stop()
        if complete:
            break
    return lm_loss_reduced, skipped_iter, mems, img_loss, txt_loss


def report_iteration_metrics(summary_writer, optimizer, lr, loss, elapsed_time, step, total_step, args, img_loss, txt_loss):
    log_string = ' iteration {:8d}/{:8d} |'.format(step, total_step)
    log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time)
    log_string += ' learning rate {:.3E} |'.format(lr)
    log_string += ' lm loss {:.6E} |'.format(loss)
    log_string += ' img loss {:.6E} |'.format(img_loss)
    log_string += ' unscaled txt loss {:.6E} |'.format(txt_loss)
    if args.fp16:
        log_string += ' loss scale {:.1f} |'.format(
            optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
    print_rank_0(log_string)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/lr', lr, step)
        summary_writer.add_scalar(f'Train/train_loss', loss, step)
        summary_writer.add_scalar(f'Train/elapsed_time', elapsed_time, step)


def report_evaluate_metrics(summary_writer, prefix, loss, ppl, step):
    string = ' validation loss at {} | '.format(prefix)
    string += 'LM loss: {:.6E} | '.format(loss)
    string += 'LM PPL: {:.6E}'.format(ppl)
    length = len(string) + 1
    print_rank_0('-' * 100)
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/valid_ppl', ppl, step)
        summary_writer.add_scalar(f'Train/valid_loss', loss, step)


def train(model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args, summary_writer=None):
    """Train the model."""
    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0
    total_img_loss = total_txt_loss = 0.0

    # Iterations.
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    mems = []
    while args.iteration < args.train_iters:

        if args.iteration % 100 == 0:
            new_loaders = detect_new_datasets(args)
            if new_loaders is not None:
                print(f'Loatding new datasets ... Now we train models on {args.train_data}.')
                train_data_iterator = iter(new_loaders[0])
                val_data_iterator = iter(new_loaders[1])
                # TODO close the original


        lm_loss, skipped_iter, mems, img_loss, txt_loss = train_step(train_data_iterator,
                                           model,
                                           optimizer,
                                           lr_scheduler,
                                           args, timers, mems)
        skipped_iters += skipped_iter
        args.iteration += 1

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()
        total_img_loss += img_loss.data.detach().float()
        total_txt_loss += txt_loss.data.detach().float()

        # Logging.
        if args.iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            # average img & txt loss
            avg_img_loss = total_img_loss.item() / args.log_interval
            avg_txt_loss = total_txt_loss.item() / args.log_interval

            elapsed_time = timers('interval time').elapsed()
            report_iteration_metrics(summary_writer, optimizer, learning_rate, avg_lm_loss,
                                    elapsed_time * 1000.0 / args.log_interval, args.iteration, args.train_iters, args,
                                    avg_img_loss, avg_txt_loss)
            total_lm_loss = 0.0
            total_img_loss = 0.0
            total_txt_loss = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(args.iteration))
                report_memory_flag = False
            if USE_TORCH_DDP:
                timers.log(['forward', 'backward', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
            else:
                timers.log(['forward', 'backward', 'allreduce', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
        # Checkpointing
        if args.save and args.save_interval and args.iteration % args.save_interval == 0:
            save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and args.iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(args.iteration)
            evaluate_and_print_results(
                prefix, val_data_iterator, model, args, timers, False, step=args.iteration, summary_writer=summary_writer)

        if args.exit_interval and args.iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, args.iteration), flush=True)
            exit()

    return args.iteration, skipped_iters


def evaluate(data_iterator, model, args, timers, verbose=False):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()
    is_sparse_raw = args.is_sparse
    args.is_sparse = 0

    total_lm_loss = 0
    mems = []
    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
            # Forward evaluation.
            lm_loss, mems, img_loss, txt_loss = forward_step(data_iterator, model, args, timers, mems=mems)

            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

            # Reduce across processes.
            if isinstance(model, DDP):
                torch.distributed.all_reduce(lm_loss.data)
                lm_loss.data = lm_loss.data / args.world_size

            total_lm_loss += lm_loss.data.detach().float().item()

    # Move model back to the train mode.
    model.train()
    args.is_sparse = is_sparse_raw

    total_lm_loss /= args.eval_iters
    return total_lm_loss


def evaluate_and_print_results(prefix, data_iterator, model,
                               args, timers, verbose=False, step=None, summary_writer=None):
    """Helper function to evaluate and dump results on screen."""
    lm_loss = evaluate(data_iterator, model, args, timers, verbose)
    lm_ppl = math.exp(min(20, lm_loss))
    report_evaluate_metrics(summary_writer, prefix, lm_loss, lm_ppl, step)

    return lm_loss


'''
    Optional DeepSpeed Activation Checkpointing features
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.

    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.

    This must be done before all the calls to mpu.model_parallel_cuda_manual_seed
    '''


def set_deepspeed_activation_checkpointing(args):
    deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if hasattr(args, "deepspeed") and args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def get_train_val_test_data(args):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, val_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        train_data, val_data, test_data = make_loaders(args)
        num_tokens = get_tokenizer().num_tokens

        before = num_tokens
        after = before
        multiple = args.make_vocab_size_divisible_by * \
                   mpu.get_model_parallel_world_size()
        while (after % multiple) != 0:
            after += 1
        print_rank_0('> padded vocab (size: {}) with {} dummy '
                     'tokens (new size: {})'.format(
                         before, after - before, after))
        token_counts = torch.cuda.LongTensor(
            [after, int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0])
    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    args.do_train = token_counts[1].item()
    args.do_valid = token_counts[2].item()
    args.do_test = token_counts[3].item()

    return train_data, val_data, test_data, num_tokens


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False
    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()
    if args.load:
        args.experiment_name = os.path.basename(os.path.normpath(args.load))
    else:
        args.experiment_name = args.experiment_name + datetime.now().strftime("%m-%d-%H-%M")
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)
    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # init tokenizer
    tokenizer = get_tokenizer(args)

    # Data stuff.
    train_data, val_data, test_data, args.vocab_size = get_train_val_test_data(args)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)

    if args.load is not None:
        if args.fast_load:
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
        else:
            with FileLock("/root/checkpoint_lock", timeout=-1):
                args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0
    torch.distributed.barrier()

    summary_writer = None
    if torch.distributed.get_rank() == 0:
        if args.finetune:
            print('Finetune CogView model')
        else:
            print('Pretrain CogView model')
        print_args(args)
        summary_writer = get_sample_writer(base=args.summary_dir, name=args.experiment_name, iteration=args.iteration)

    # Resume data loader if necessary.
    if args.resume_dataloader:
        if train_data is not None:
            train_data.batch_sampler.start_iter = args.iteration % \
                                                  len(train_data)
        if val_data is not None:
            start_iter_val = (args.train_iters // args.save_interval) * \
                             args.eval_interval
            val_data.batch_sampler.start_iter = start_iter_val % \
                                                len(val_data)
    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None

    # TODO: figure out how to properly set this especially when resuming training
    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            with ExitStack() as stack:
                def save_on_exit(args_, model_, optimizer_, lr_scheduler_):
                    save_checkpoint(args_.iteration, model_, optimizer_, lr_scheduler_, args_)
                # stack.callback(save_on_exit, args, model, optimizer, lr_scheduler)
                iteration, skipped = train(model, optimizer,
                                           lr_scheduler,
                                           train_data_iterator,
                                           val_data_iterator,
                                           timers, args, summary_writer=summary_writer)

        if args.do_valid:
            prefix = 'the end of training for val data'
            val_loss = evaluate_and_print_results(prefix, val_data_iterator,
                                                  model, args, timers, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

    if test_data is not None:
        test_data_iterator = iter(test_data)
    else:
        test_data_iterator = None

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, test_data_iterator,
                                   model, args, timers, True)


if __name__ == "__main__":
    main()
