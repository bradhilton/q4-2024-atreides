#!/bin/bash

if [ -z "$1" ]; then
    echo "Please provide a name argument"
    exit 1
fi

NAME=$1

jupyter nbconvert experiments/002-llama3.ipynb --to python --TagRemovePreprocessor.remove_cell_tags='["magic"]'
uv run torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) experiments/002-llama3.py --name $NAME
