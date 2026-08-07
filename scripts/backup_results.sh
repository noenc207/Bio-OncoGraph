#!/bin/bash
set -e

RESULTS_DIR="results"
BACKUP_DIR="backups"
mkdir -p "$BACKUP_DIR"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "No results directory found to backup."
    exit 0
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/results_backup_${TIMESTAMP}.tar.gz"

echo "Backing up $RESULTS_DIR to $BACKUP_FILE..."
tar -czf "$BACKUP_FILE" "$RESULTS_DIR"

echo "Backup completed: $BACKUP_FILE"
