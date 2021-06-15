#!/bin/bash

CHECKPOINT_PATH=pretrained/cogview/cogview-sr
NLAYERS=48
NHIDDEN=2560
NATT=40
MAXSEQLEN=1345
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MPSIZE=1

#SAMPLING ARGS
TEMP=1.02
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=200
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
 
MASTER_PORT=${MASTER_PORT} python generate_samples.py \
       --deepspeed \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --max-position-embeddings 1089 \
       --fp16 \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       --img-tokenizer-path pretrained/vqvae/vqvae_hard_biggerset_011.pt \
       --query-window 64 \
       --key-window-times 4 \
       --num-pivot 256 \
       --is-sparse 0 \
       --max-position-embeddings-finetune $MAXSEQLEN \
       --generation-task "super-resolution" \
       --input-source interactive \
       --output-path samples_sr \
       --debug \
       --device 0 \
       $@


