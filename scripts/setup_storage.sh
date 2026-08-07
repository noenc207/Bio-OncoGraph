#!/bin/bash

# setup_storage.sh
# Script to create the directory structure for Bio-OncoGraph.
# Includes robust error checking.

set -e
set -u
set -o pipefail

# Default base directory is the project root (assuming script is run from project root or inside scripts/)
# We'll use the directory where the script is located to find the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define directories to create relative to project root
declare -a DIRS=(
    "datasets/raw"
    "datasets/processed"
    "datasets/patches"
    "datasets/graphs"
    "datasets/features"
    "datasets/embeddings"
    "datasets/clinical"
    "datasets/survival"
    "datasets/genomics"
    "datasets/splits"
    "datasets/cache"
    "weights"
    "checkpoints"
    "runs"
    "outputs"
    "artifacts"
    "logs"
    "reports"
)

echo "Setting up storage directories in ${PROJECT_ROOT}..."

for DIR in "${DIRS[@]}"; do
    TARGET_DIR="${PROJECT_ROOT}/${DIR}"
    if [ ! -d "${TARGET_DIR}" ]; then
        if mkdir -p "${TARGET_DIR}"; then
            echo "Created directory: ${TARGET_DIR}"
        else
            echo "Error: Failed to create directory ${TARGET_DIR}" >&2
            exit 1
        fi
    else
        echo "Directory already exists: ${TARGET_DIR}"
    fi
done

echo "Storage setup completed successfully."
exit 0
