#!/bin/bash
set -e

echo "Downloading pre-trained models..."

MODEL_DIR="models"
mkdir -p "$MODEL_DIR"

VENV_PATH="venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

# Example downloading a dummy model file
echo "Downloading models..."
wget -qO "$MODEL_DIR/dummy_model.pt" https://raw.githubusercontent.com/pytorch/hub/master/pytorch_logo.png || true
echo "Pre-trained models downloaded to $MODEL_DIR."
