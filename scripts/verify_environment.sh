#!/bin/bash
set -e

echo "Verifying environment..."

# Check NVIDIA drivers
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi could not be found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

echo "NVIDIA GPU Information:"
nvidia-smi

# Check Python environment
VENV_PATH="venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found. Please run install_dependencies.sh first."
    exit 1
fi

source "$VENV_PATH/bin/activate"

# Check OpenSlide
python3 -c "import openslide; print(f'OpenSlide version: {openslide.__version__}')" || { echo "Error: openslide-python not working properly."; exit 1; }

# Check PyTorch CUDA availability
python3 -c "import torch; print(f'PyTorch version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')" || { echo "Error: PyTorch not working properly."; exit 1; }

echo "Environment verification completed successfully."
