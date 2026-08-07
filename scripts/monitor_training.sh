#!/bin/bash
set -e

LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
    echo "No logs directory found."
    exit 1
fi

LATEST_LOG=$(ls -t "$LOG_DIR"/training_*.log "$LOG_DIR"/resume_training_*.log 2>/dev/null | head -n 1)

if [ -z "$LATEST_LOG" ]; then
    echo "No log files found."
    exit 1
fi

echo "Monitoring latest log file: $LATEST_LOG"
echo "Press Ctrl+C to exit."

tail -f "$LATEST_LOG"
