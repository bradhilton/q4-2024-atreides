#!/bin/bash
# Enable faster downloads from Hugging Face
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download the model
CHECKPOINT_DIR=$(huggingface-cli download NousResearch/Hermes-2-Theta-Llama-3-8B)

# Run the training script
tune run experiments/recipes/full_finetune.py --config ./experiments/recipes/configs/llama3_1/8B_full.yaml checkpointer.checkpoint_dir=$CHECKPOINT_DIR
