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

"""Transformer."""

import math
import random

import torch
import torch.nn.init as init
from apex.normalization.fused_layer_norm import FusedLayerNorm #as LayerNorm

from .initialize import get_model_parallel_world_size
from .layers import ColumnParallelLinear
from .layers import RowParallelLinear
from .mappings import gather_from_model_parallel_region

import deepspeed

from .random import checkpoint
from .random import get_cuda_rng_tracker

from .utils import divide
from .utils import split_tensor_along_last_dim
import torch.distributed as dist


class LayerNorm(FusedLayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return super().forward(x / (x.abs().max().detach()/8))

class GPT2ParallelSelfAttention(torch.nn.Module):
    """Parallel self-attention layer for GPT2.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence length, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """
    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, output_layer_init_method=None, query_window=128, key_window_times=6):
        super(GPT2ParallelSelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)
        self.query_window = query_window
        self.key_window_times = key_window_times

        # Strided linear layer.
        self.query_key_value = ColumnParallelLinear(hidden_size, 3*hidden_size,
                                                    stride=3,
                                                    gather_output=False,
                                                    init_method=init_method)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)


    def forward(self, hidden_states, ltor_mask, pivot_idx=None, is_sparse=0, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Attention heads. [b, s, hp]
        query_length = hidden_states.size(1)

        if mem is None:
            mixed_x_layer = self.query_key_value(hidden_states)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            cat = torch.cat((mem, hidden_states), 1)
            mixed_x_layer = self.query_key_value(cat)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
            mixed_query_layer = mixed_query_layer[:, -query_length:]

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # =====================   Core Attention Code  ======================== #
        if is_sparse == 1:
            context_layer = sparse_attention(query_layer, key_layer, value_layer, pivot_idx, ltor_mask, self.query_window, self.key_window_times, self.attention_dropout)
        elif is_sparse == 2:
            context_layer = sparse_attention_inference(query_layer, key_layer, value_layer, pivot_idx)
        else:
            context_layer = standard_attention(query_layer, key_layer, value_layer, ltor_mask, self.attention_dropout)
        
        # ===================== END OF BLOCK ======================= #

        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


@torch.jit.script
def gelu_impl(x):
     """OpenAI's gelu implementation."""
     return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                        (1.0 + 0.044715 * x * x)))

def gelu(x): 
    return gelu_impl(x)

@torch.jit.script
def elu1_impl(x):
     """OpenAI's gelu implementation."""
     return torch.nn.functional.elu(x) + 1.

def elu1(x):
    return elu1_impl(x)

class GPT2ParallelMLP(torch.nn.Module):
    """MLP for GPT2.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    def __init__(self, hidden_size, output_dropout_prob, init_method,
                 output_layer_init_method=None):
        super(GPT2ParallelMLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4*hidden_size,
                                                  gather_output=False,
                                                  init_method=init_method)
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4*hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method)
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class GPT2ParallelTransformerLayer(torch.nn.Module):
    """A single layer transformer for GPT2.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None,
                 query_window=128,
                 key_window_times=6,
                 scale_normalization=True
                 ):
        super(GPT2ParallelTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)


        # Self attention.
        self.attention = GPT2ParallelSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            query_window=query_window,
            key_window_times=key_window_times)

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size,
                                                  eps=layernorm_epsilon)
        self.scale_normalization = scale_normalization
        if scale_normalization:
            self.third_layernorm = LayerNorm(hidden_size,
                                                    eps=layernorm_epsilon)
            self.fourth_layernorm = LayerNorm(hidden_size,
                                                    eps=layernorm_epsilon)

        # MLP
        self.mlp = GPT2ParallelMLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(self, hidden_states, ltor_mask, pivot_idx=None, is_sparse=0, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output1 = self.input_layernorm(hidden_states)
        mem = self.input_layernorm(mem) if mem is not None else None
        # Self attention.
        attention_output = self.attention(layernorm_output1, ltor_mask, pivot_idx, is_sparse, mem)

        # Third LayerNorm
        if self.scale_normalization:
            attention_output = self.third_layernorm(attention_output)

        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Fourth LayerNorm
        if self.scale_normalization:
            mlp_output = self.fourth_layernorm(mlp_output)

        # Second residual connection.
        output = layernorm_input + mlp_output

        return output

def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GPT2ParallelTransformer(torch.nn.Module):
    """GPT-2 transformer.

    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """
    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 max_memory_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 use_scaled_init_for_output_weights=True,
                 query_window=128,
                 key_window_times=6,
                 num_pivot=768
                 ):
        super(GPT2ParallelTransformer, self).__init__()
        # Store activation checkpoiting flag.
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_memory_length = max_memory_length
        self.max_sequence_length = max_sequence_length

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(init_method_std,
                                                      num_layers)
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(max_sequence_length,
                                                        hidden_size)
        # Initialize the position embeddings.
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        # TODO: after testing, this is not useful.
        # self.img_type_embeddings = torch.nn.Parameter(torch.Tensor(64, hidden_size)) 
        # torch.nn.init.normal_(self.img_type_embeddings, mean=0.0, std=init_method_std)
        # self.txt_type_embeddings = torch.nn.Parameter(torch.Tensor(hidden_size)) 
        # torch.nn.init.normal_(self.txt_type_embeddings, mean=0.0, std=init_method_std)


        def get_layer(layer_id):
            return GPT2ParallelTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                unscaled_init_method(init_method_std),
                output_layer_init_method=output_layer_init_method,
                query_window=query_window,
                key_window_times=key_window_times,
                scale_normalization=True
                )

        self.query_window = query_window
        self.key_window_times = key_window_times
        self.num_pivot = num_pivot

        # Transformer layers.
        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(num_layers)])

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint
        self.rmask = None

    def forward(self, hidden_states, position_ids, attention_mask, txt_indices_bool, img_indices_bool, is_sparse=0, *mems):

        batch_size, query_length = hidden_states.size()[:2]
        memory_length = mems[0].size(1) if mems else 0
        key_length = query_length + memory_length

        if isinstance(attention_mask, int) or attention_mask.numel() == 1:
            # if given a int "sep", means the seperation of full attention part and single direction part
            # attention mask is the beginning postion of B region, \in [0, query_len)
            sep = attention_mask
            # conventional transformer
            def build_mask_matrix(query_length, key_length, sep):
                m = torch.ones((1, query_length, key_length), device=hidden_states.device, dtype=hidden_states.dtype)
                assert query_length <= key_length
                m[0, :, -query_length:] = torch.tril(m[0, :, -query_length:])
                m[0, :, :sep + (key_length - query_length)] = 1
                m = m.unsqueeze(1)
                return m
            attention_mask = build_mask_matrix(query_length, key_length, sep)

        if is_sparse == 1 and (self.rmask is None):
            w, times = self.query_window, self.key_window_times
            g = key_length // w
            tmp = torch.ones((g-times+1, w , w), device=hidden_states.device, dtype=hidden_states.dtype)
            tmp = torch.tril(1 - torch.block_diag(*tmp))
            self.rmask = torch.nn.functional.pad(tmp, (0, (times-1)*w, (times-1)*w, 0)) # pad (left, right, top, bottom)  

        if is_sparse == 2:
            left_boundary = max(0, key_length - self.key_window_times * self.query_window)
            window_idx = torch.arange(left_boundary, key_length, device=hidden_states.device, dtype=torch.long).expand(batch_size, -1)
        elif is_sparse == 1:
            left_boundary = key_length
            num_pivot = self.num_pivot
                
        # =====================   Image & Text Type Embedding   ======================== #
        # TODO: after testing, this is not useful.
        # extend_len = (key_length + 63) // 64
        # hidden_states = hidden_states + txt_indices_bool.unsqueeze(-1) * self.txt_type_embeddings.view(1, 1, -1) + \
        #     img_indices_bool.unsqueeze(-1) * self.img_type_embeddings.expand(extend_len, 64, -1).reshape(extend_len * 64, -1)[memory_length: key_length]
        # ===================== END OF BLOCK ======================= #

        if is_sparse: # 1 or 2                
            # select out the real indices for sampling
            img_indices = [img_indices_bool[i][:left_boundary].nonzero(as_tuple=False).view(-1) for i in range(batch_size)]
            txt_indices = [txt_indices_bool[i][:left_boundary].nonzero(as_tuple=False).view(-1) for i in range(batch_size)]
        
        if is_sparse == 2:
            ratio = self.num_pivot / self.max_sequence_length 
            max_text_num = max(len(text_idx) for text_idx in txt_indices)
            num_pivot = max_text_num + int((left_boundary - max_text_num) * ratio)

        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        if self.max_memory_length > 0:
            mem_layers = [hidden_states.detach()]
        else:
            mem_layers = []
        def custom(start, end):
            def custom_forward(*inputs):
                layers_ = self.layers[start:end]
                x_, inputs = inputs[0], inputs[1:]
                    
                if is_sparse > 0:
                    inputs, mems_ = inputs[:3], inputs[3:]
                else:
                    inputs, mems_ = inputs[:1], inputs[1:]

                for i, layer in enumerate(layers_):
                    mem_i_ = mems_[i] if mems_ else None
                    x_ = layer(x_, *inputs, mem=mem_i_)
                    if self.max_memory_length > 0:
                        mem_layers.append(x_.detach())
                return x_
            return custom_forward

        attention_mask_saved = attention_mask
        
        if self.checkpoint_activations:
            l = 0
            num_layers = len(self.layers)
            chunk_length = self.checkpoint_num_layers
            while l < num_layers:
                if is_sparse > 0:
                    # =====================   Pivot Mask   ======================== #
                    pivot_idx = torch.stack([
                        torch.cat((
                            text_idx,
                            img_indices[i][
                                torch.tensor(random.sample(range(len(img_indices[i])), k=num_pivot - len(text_idx)), dtype=torch.long, device=text_idx.device)
                            ]
                        ), dim=0)
                        for i, text_idx in enumerate(txt_indices)
                    ])
                    if is_sparse == 1: # sparse training
                        assert key_length == query_length
                        b, s = batch_size, key_length
                        pivot_attention_mask = self.rmask.expand(b, s, s).gather(dim=-1, index=pivot_idx.unsqueeze(1).expand(b, s, self.num_pivot))
                        args = [hidden_states, pivot_attention_mask, pivot_idx, torch.tensor(is_sparse)]
                    elif is_sparse == 2: # sparse inference
                        pw_idx = torch.cat((pivot_idx, window_idx), dim=-1)
                        args = [hidden_states, attention_mask_saved, pw_idx, torch.tensor(is_sparse)]
                    else:
                        raise NotImplementedError
                    # ===================== END OF BLOCK ======================= #
                else:
                    args = [hidden_states, attention_mask_saved]

                if mems:
                    args += mems[l: l + chunk_length]

                hidden_states = checkpoint(custom(l, l + chunk_length), *args)
                l += chunk_length
        else:
            assert is_sparse != 1, 'Please use checkpoint_activations for sparse attention training.'
            for i, layer in enumerate(self.layers):
                if is_sparse == 0:
                    args = [hidden_states, attention_mask_saved]
                elif is_sparse == 2:
                    pivot_idx = torch.stack([
                        torch.cat((
                            text_idx,
                            img_indices[i][
                                torch.tensor(random.sample(range(len(img_indices[i])), k=num_pivot - len(text_idx)), dtype=torch.long, device=text_idx.device)
                            ]
                        ), dim=0)
                        for i, text_idx in enumerate(txt_indices)
                    ])
                    pw_idx = torch.cat((pivot_idx, window_idx), dim=-1)
                    args = [hidden_states, attention_mask_saved, pw_idx, torch.tensor(is_sparse)]

                mem_i = mems[i] if mems else None
                hidden_states = layer(*args, mem=mem_i)
                if self.max_memory_length > 0:
                    mem_layers.append(hidden_states.detach())

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if self.max_memory_length > 0:
            mem_layers = self.update_mems(mem_layers, mems)

        return (output, *mem_layers)

    def update_mems(self, hiddens, mems):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = min(self.max_memory_length, memory_length + query_length)
        new_mems = []
        with torch.no_grad():
            for i in range(len(hiddens)):
                if new_memory_length <= query_length:
                    new_mems.append(hiddens[i][:, -new_memory_length:])
                else:
                    new_mems.append(torch.cat((mems[i][:, -new_memory_length+query_length:], hiddens[i]), dim=1))
        return new_mems
        

def _chunk(x, w, times):
    '''convert into overlapping chunkings. Chunk size = times * w, overlap size = w
    Args:
        x: [b, np, s, hn]
        ...
    '''
    s = x.size(2)
    # x pad to [b, np, s+xx to k*w + w*(times-1), hn]
    assert s % w == 0
    npad = (times-1) * w
    x = torch.nn.functional.pad(x, (0, 0, npad, 0), value=0)

    x = x.view(x.size(0), x.size(1),  x.size(2) // w, w, x.size(3))

    chunk_size = list(x.size())
    chunk_stride = list(x.stride())

    chunk_size[2] = chunk_size[2] - times + 1

    chunk_size[3] = w * times

    return x.as_strided(size=chunk_size, stride=chunk_stride)

def standard_attention(query_layer, key_layer, value_layer, attention_mask, attention_dropout=None):
    # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training. 
    # The implementation in the paper can be done very easily, if you really need it to train very deep transformers. 

    if len(attention_mask.shape) == 3:
        attention_mask = attention_mask.unsqueeze(1)
    # Raw attention scores. [b, np, s, s]
    attention_scores = torch.matmul(query_layer / math.sqrt(query_layer.shape[-1]), key_layer.transpose(-1, -2))

    # Apply the left to right attention mask.
    attention_scores = torch.mul(attention_scores, attention_mask) - \
                    10000.0 * (1.0 - attention_mask)
    # Attention probabilities. [b, np, s, s]
    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs = attention_dropout(attention_probs)
    # Context layer.
    # [b, np, s, hn]
    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer

def sparse_attention(q, k, v, pivot_idx, pivot_attention_mask, query_window=128, key_window_times=6, attention_dropout=None):
    ''' Sparse Attention
    Args:
        q, k, v: inputs, [b, num_heads, s, hn], k is padded to n * query_window
        pivot_idx: [b, num_pivots]
        pivot_attention_mask: [b, s, num_pivots]
        query_window: .
        key_window_times: key_window = query_window * key_window_times
    '''

    b, n_head, s, hn = q.shape
    b, n_piv = pivot_idx.shape
    w = query_window

    pivot_idx_dummy = pivot_idx.view(b, 1, n_piv, 1).expand(b, n_head, n_piv, hn)
    # =====================   Pivot Attention   ======================== #
    pivot_k, pivot_v = torch.gather(k, 2, pivot_idx_dummy), torch.gather(v, 2, pivot_idx_dummy)
    attention_scores = torch.matmul(q, pivot_k.transpose(-1, -2))
    pivot_attention_mask = pivot_attention_mask.unsqueeze(1)

    attention_scores_pivot = torch.mul(attention_scores, pivot_attention_mask / math.sqrt(hn)) - 10000.0 * (1.0 - pivot_attention_mask)

    attention_scores_pivot = attention_scores_pivot + math.log(s // n_piv)
    # =====================   Window Attention   ======================= #
    window_k = _chunk(k, query_window, key_window_times)
    window_v = _chunk(v, query_window, key_window_times)
    # window_k [b, n_head, s // w up int, w*times, hn]

    if s % w == 0: # training # TODO args check
        assert k.shape[2] == s
        assert window_k.shape[2] == s // w
        window_q = q.view(b, n_head, s // w, w, hn)        
        attention_scores = torch.matmul(window_q, window_k.transpose(-1, -2))
        window_attention_mask = torch.ones((w, w * key_window_times), dtype=attention_scores.dtype, device=q.device).tril_(diagonal=w * (key_window_times - 1))
        attention_scores_window = torch.mul(attention_scores, window_attention_mask / math.sqrt(hn)) - 10000.0 * (1.0 - window_attention_mask)
        for t in range(1, key_window_times):
            attention_scores_window[:, :, t - 1, :, :w * key_window_times - w * t] -= 10000.0
    else: 
        raise ValueError('The seq_len must be exactly divided by window_size.')
    # =====================   Joint Softmax   ======================= #
    attention_scores_window = attention_scores_window.view(b, n_head, s, w * key_window_times)
    attention_scores = torch.cat((attention_scores_pivot, attention_scores_window), dim=-1)
    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs = attention_dropout(attention_probs)

    context_layer = torch.matmul(attention_probs[..., :-w * key_window_times], pivot_v) + torch.einsum('bcgwk,bcgkh->bcgwh', attention_probs[..., -w * key_window_times:].view(b, n_head, s // w, w, w * key_window_times), window_v).view(b, n_head, s, hn)

    return context_layer

def sparse_attention_inference(q, k, v, pivot_and_window_idx, **kwargs):
    '''the inference process of sparse attention.
    The Qs are in the same block, but seq_len mod window size might != 0.

    The Qs are the final tokens of Ks. the pivot_and_window_idx[-query_len] are Qs.

    '''
    b, n_head, sq, hn = q.shape
    sk = k.shape[2]
    _b, n_piv = pivot_and_window_idx.shape

    pivot_and_window_idx_dummy = pivot_and_window_idx.view(b, 1, n_piv, 1).expand(b, n_head, n_piv, hn)
    pivot_k, pivot_v = torch.gather(k, 2, pivot_and_window_idx_dummy), torch.gather(v, 2, pivot_and_window_idx_dummy)
    attention_scores = torch.matmul(q / math.sqrt(hn), pivot_k.transpose(-1, -2))
    if sq > 1:
        query_part_scores = attention_scores[:, :, -sq:, -sq:]
        m = torch.ones((sq, sq), device=q.device, dtype=q.dtype) * -10000.
        m.triu_(diagonal=1)
        query_part_scores += m

    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

    context_layer = torch.matmul(attention_probs, pivot_v) 
    return context_layer


def test_sparse_attention():       

    s, w, times = 4096 + 128, 128, 2
    num_pivot = 768
    b = 2
    g = s // w

    q, k, v = raw = torch.rand(3, b, 16, s, 64, dtype=torch.float, device='cuda', requires_grad=True)
    q1, k1, v1 = raw1 = torch.tensor(raw.cpu().detach().numpy(), dtype=torch.float, device='cuda', requires_grad=True)
    txt_indices = [torch.arange(0, 128, dtype=torch.long, device='cuda'), torch.arange(0, 22, dtype=torch.long, device='cuda')]
    img_indices = [torch.arange(128, s, dtype=torch.long, device='cuda'), torch.arange(22, s, dtype=torch.long, device='cuda')]

    pivot_idx = torch.stack([
        torch.cat((
            text_idx,
            img_indices[i][
                torch.tensor(random.sample(range(len(img_indices[i]) - times*w),  k=num_pivot - len(text_idx)), dtype=torch.long, device=text_idx.device)
            ]
        ), dim=0)
        for i, text_idx in enumerate(txt_indices)
    ]) # -times * w to verify inference

    tmp = torch.ones((g-times+1, w , w), device='cuda', dtype=torch.long)
    tmp = torch.tril(1 - torch.block_diag(*tmp))
    rmask = torch.nn.functional.pad(tmp, (0, (times-1)*w, (times-1)*w, 0)) # pad (left, right, top, bottom)

    pivot_attention_mask = rmask.expand(b, s, s).gather(dim=-1, index=pivot_idx.unsqueeze(1).expand(b, s, num_pivot))

    real_mask = torch.ones((b, s, s), device='cuda', dtype=torch.long) - rmask
    for i in range(b):
        real_mask[i][:, pivot_idx[i]] = 1
        real_mask[i].tril_()

    # test inference

    # q_part = q[..., -1:, :]
    # r0 = standard_attention(q, k, v, real_mask)
    # r0 = r0[..., -1:, :]
    # pw_idx = torch.cat((pivot_idx, torch.arange(s-times*w, s, device='cuda', dtype=torch.long).expand(b, -1)), dim=-1)

    # r1 = sparse_attention_inference(q_part, k, v, pw_idx)
    # print(( (r1-r0).abs() / (r1.abs()+r0.abs())).max())

    import time

    r0 = standard_attention(q1, k1, v1, real_mask)
    torch.cuda.synchronize()
    t0 = time.time()
    r1 = standard_attention(q1, k1, v1, real_mask)
    torch.cuda.synchronize()
    t1 = time.time()
    r2 = sparse_attention(q, k, v, pivot_idx, pivot_attention_mask, w, times)
    torch.cuda.synchronize()
    t2 = time.time()
    print('times: standard ', t1-t0, ' sparse ', t2-t1)

    print(( (r1-r2).abs() / (r1.abs()+r2.abs())).max())

    raw.retain_grad()
    l2 = r2.mean()
    l1 = r1.mean()
    l2.backward()
    l1.backward()

    g1 = raw1.grad
    g2 = raw.grad
    print( (g1-g2).abs().max())

    # import pdb; pdb.set_trace()
