#!/bin/bash
set -e

echo "Validating release..."

echo "Running Bandit for security analysis..."
bandit -r bio_oncograph/ -c pyproject.toml

echo "Running pip-audit for dependency vulnerability scanning..."
pip-audit -r requirements.txt -r requirements-dev.txt

echo "Checking license compatibility..."
pip-licenses --format=markdown --with-system --output-file=licenses.md

echo "Release validation complete."
