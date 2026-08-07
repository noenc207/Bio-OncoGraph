#!/bin/bash
set -e

echo "Preparing server for Bio-OncoGraph on Ubuntu 24.04..."

# Update package lists
sudo apt-get update -y
sudo apt-get upgrade -y

# Install essential tools
sudo apt-get install -y build-essential curl wget git tmux htop software-properties-common

# Install OpenSlide
sudo apt-get install -y openslide-tools libopenslide-dev

# Install Python and pip (Ubuntu 24.04 defaults to Python 3.12)
sudo apt-get install -y python3 python3-pip python3-venv python3-dev

echo "Server preparation completed successfully."
