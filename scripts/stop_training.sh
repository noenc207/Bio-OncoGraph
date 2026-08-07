#!/bin/bash
set -e

PID_FILE="training.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No training.pid file found. Is training running?"
    
    PID=$(pgrep -f "python3 main.py" || true)
    if [ -z "$PID" ]; then
        echo "No training process found."
        exit 0
    fi
else
    PID=$(cat "$PID_FILE")
fi

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping training process $PID..."
    kill "$PID"
    rm -f "$PID_FILE"
    echo "Process stopped."
else
    echo "Process $PID is not running."
    rm -f "$PID_FILE"
fi
