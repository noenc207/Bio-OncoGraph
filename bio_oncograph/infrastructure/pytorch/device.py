"""PyTorch device management module."""
import torch


def get_device() -> torch.device:
    """Gets the optimal PyTorch device available.

    Returns:
        torch.device: The selected PyTorch device ('cuda' if available, otherwise 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_available_devices() -> list[torch.device]:
    """Gets a list of all available PyTorch devices.
    
    Returns:
        list[torch.device]: A list containing the CPU device and all available CUDA devices.
    """
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f"cuda:{i}"))
    return devices
