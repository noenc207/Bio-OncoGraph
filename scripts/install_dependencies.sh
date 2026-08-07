#!/bin/bash
set -e

echo "Installing Python dependencies..."

# Define virtual environment path
VENV_PATH="venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (assuming CUDA 12.1 for modern setups)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install OpenSlide Python wrapper and other common ML libraries
pip install openslide-python numpy pandas scikit-learn transformers huggingface-hub wandb tqdm matplotlib seaborn

# Install any requirements.txt if present
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "Dependencies installed successfully."
