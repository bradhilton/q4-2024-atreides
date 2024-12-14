#!/bin/bash
uv run fastapi run --port=2218 --workers=$(nproc) ./experiments/lib/clue.py