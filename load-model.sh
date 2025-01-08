#!/bin/bash

# Check if model name is provided as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 bert-base"
    exit 1
fi

MODEL_NAME=$1
SOURCE_PATH="gs://atreides/experiments/models/${MODEL_NAME}"
DEST_PATH="./experiments/models/${MODEL_NAME}"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_PATH"

# Sync the model files
echo "Syncing model from ${SOURCE_PATH} to ${DEST_PATH}"
gsutil -m rsync -r "$SOURCE_PATH" "$DEST_PATH"

if [ $? -eq 0 ]; then
    echo "Model sync completed successfully"
else
    echo "Error: Failed to sync model"
    exit 1
fi 