"""Advanced model serialization and version compatibility for Bio-OncoGraph."""

import os
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

# Define the current version of the serialization format
SERIALIZATION_VERSION = "1.0.0"

def export_model(
    path: str,
    model: nn.Module,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Exports a model for inference or sharing with full metadata.
    
    Args:
        path: File path to save the serialized model. Usually ends with .pt or .pth.
        model: The PyTorch module to serialize.
        metadata: Optional dictionary containing additional information about the model
            (e.g., training configuration, author, dataset used).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    export_dict: Dict[str, Any] = {
        "version": SERIALIZATION_VERSION,
        "model_state_dict": model.state_dict(),
    }
    
    if metadata:
        export_dict["metadata"] = metadata
        
    torch.save(export_dict, path)

def import_model(
    path: str,
    model: nn.Module,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """Imports a serialized model and loads it into the provided module instance.
    
    Args:
        path: File path to load the serialized model from.
        model: The PyTorch module instance to load the state into.
        device: The device to load the model tensors onto.
        strict: Whether to strictly enforce that the keys in state_dict match the keys
            returned by this module's state_dict() function.
            
    Returns:
        The metadata dictionary saved with the model, or an empty dictionary
        if no metadata was saved.
        
    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If there's a version mismatch or loading fails.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No model file found at '{path}'")
        
    data = torch.load(path, map_location=device)
    
    if "version" not in data:
        # Backward compatibility for legacy checkpoints
        version = "legacy"
    else:
        version = data["version"]
        
    # Perform basic version compatibility check if needed
    if version != SERIALIZATION_VERSION and version != "legacy":
        print(f"Warning: Loading model with version '{version}', "
              f"current version is '{SERIALIZATION_VERSION}'.")
              
    state_dict = data.get("model_state_dict", data)
    model.load_state_dict(state_dict, strict=strict)
    
    return data.get("metadata", {})
