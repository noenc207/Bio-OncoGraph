#!/bin/bash
set -e

echo "Exporting results..."

RESULTS_DIR="results"
EXPORT_DIR="exports"
mkdir -p "$EXPORT_DIR"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "No results directory found."
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPORT_FILE="$EXPORT_DIR/final_results_${TIMESTAMP}.zip"

sudo apt-get install -y zip >/dev/null 2>&1 || true

echo "Zipping results to $EXPORT_FILE..."
cd "$RESULTS_DIR"
zip -rq "../$EXPORT_FILE" .
cd ..

echo "Results exported successfully to $EXPORT_FILE."
