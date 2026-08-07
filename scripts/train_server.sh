#!/bin/bash
set -e

echo "Starting training server..."

VENV_PATH="venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found."
    exit 1
fi

source "$VENV_PATH/bin/activate"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

TRAIN_SCRIPT="main.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Warning: $TRAIN_SCRIPT not found. Creating a dummy training script."
    echo "import time; print('Training started'); time.sleep(60); print('Training finished')" > "$TRAIN_SCRIPT"
fi

echo "Running training script in the background. Logging to $LOG_FILE"
nohup python3 "$TRAIN_SCRIPT" --mode train > "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > training.pid

echo "Training started with PID $PID."
