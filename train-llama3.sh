#!/bin/bash
jupyter nbconvert experiments/02-llama3.ipynb --to python --TagRemovePreprocessor.remove_cell_tags='["magic"]'
uv run torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) experiments/02-llama3.py