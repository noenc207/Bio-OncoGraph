# Server Setup Guide

## Operating System
- **Ubuntu 24.04 LTS**: Ensure the base system is updated and upgraded.
  ```bash
  sudo apt update && sudo apt upgrade -y
  ```

## CUDA Toolkit
- Install the appropriate NVIDIA driver and CUDA toolkit.
  ```bash
  sudo apt install nvidia-driver-535
  sudo apt install nvidia-cuda-toolkit
  ```

## cuDNN (CUDA Deep Neural Network library)
- Download and install cuDNN corresponding to the installed CUDA version from the NVIDIA Developer website.
  ```bash
  # Example for local installation
  sudo dpkg -i cudnn-local-repo-ubuntu2404-8.x.x.x_1.0-1_amd64.deb
  sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get install libcudnn8 libcudnn8-dev
  ```

## OpenSlide
- Install OpenSlide for reading whole-slide images (WSIs).
  ```bash
  sudo apt install openslide-tools
  sudo apt install python3-openslide
  ```
- Install the Python wrapper:
  ```bash
  pip install openslide-python
  ```
