#!/bin/bash
jupyter nbconvert experiments/02-llama3.ipynb --to python --TagRemovePreprocessor.remove_cell_tags='["magic"]'
uv run torchrun --standalone --nproc_per_node=1 experiments/02-llama3.py