#!/bin/bash
set -e

echo "Resuming training..."

VENV_PATH="venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found."
    exit 1
fi

source "$VENV_PATH/bin/activate"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/resume_training_${TIMESTAMP}.log"

TRAIN_SCRIPT="main.py"
CHECKPOINT_DIR="checkpoints"

if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A $CHECKPOINT_DIR)" ]; then
    echo "Error: No checkpoints found to resume from in $CHECKPOINT_DIR."
    exit 1
fi

LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR" | head -n 1)

echo "Resuming from checkpoint $LATEST_CHECKPOINT"
nohup python3 "$TRAIN_SCRIPT" --mode train --resume "$CHECKPOINT_DIR/$LATEST_CHECKPOINT" > "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > training.pid

echo "Training resumed with PID $PID."
