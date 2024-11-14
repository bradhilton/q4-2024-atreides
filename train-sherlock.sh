#!/bin/bash
# Enable faster downloads from Hugging Face
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download the model
CHECKPOINT_DIR=$(huggingface-cli download NousResearch/Hermes-2-Theta-Llama-3-8B)

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)

# Run the training script
tune run \
    --nnodes 1 \
    --nproc_per_node $NUM_GPUS \
    experiments/lib/recipes/full_finetune.py \
    --config ./experiments/lib/recipes/configs/llama3_1/8B_full.yaml \
    checkpointer.checkpoint_dir=$CHECKPOINT_DIR
