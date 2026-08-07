#!/bin/bash
set -e

LOG_DIR="logs"
COLLECT_DIR="collected_logs"
mkdir -p "$COLLECT_DIR"

if [ ! -d "$LOG_DIR" ]; then
    echo "No logs directory found to collect."
    exit 0
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_FILE="$COLLECT_DIR/logs_archive_${TIMESTAMP}.tar.gz"

echo "Collecting logs into $ARCHIVE_FILE..."
tar -czf "$ARCHIVE_FILE" "$LOG_DIR"

echo "Logs collected: $ARCHIVE_FILE"
