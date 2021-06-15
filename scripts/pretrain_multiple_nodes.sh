#! /bin/bash

# Change for multinode config

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_GID_INDEX=3 NCCL_NET_GDR_LEVEL=0"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
# OPTIONS_NCCL=""
# HOST_FILE_PATH="hostfile_single"


config_json="$script_dir/ds_config_zero.json"
gpt_options=" \
       --experiment-name cogview-ali_fashion_tutorial-12-1024-16 \
       --img-tokenizer-num-tokens 8192 \
       --dataset-type TokenizedDataset \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 12 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --save $main_dir/data/checkpoints \
       --train-iters 40000 \
       --resume-dataloader \
       --train-data ./data/ali_vqvae_hard_biggerset_011.lmdb \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .1 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --max-position-embeddings 1089 \
       --max-memory-length 0 \
       --fp16 \
"


gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
