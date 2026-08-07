#!/bin/bash
set -e

echo "Publishing to PyPI..."
python -m build
python -m twine upload dist/*
echo "Publish complete."
