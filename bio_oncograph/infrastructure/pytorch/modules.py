"""PyTorch module base classes for Bio-OncoGraph.

This module provides abstract wrappers for torch.nn.Module that integrate
with the framework's configuration system.
"""
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

class ConfigurableModule(nn.Module):
    """Base class for modules configured via dictionaries.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary used to initialize the module.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the ConfigurableModule.
        
        Args:
            config: A dictionary containing configuration parameters.
        """
        super().__init__()
        self.config = config
        
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration dictionary.
        
        Returns:
            The configuration dictionary.
        """
        return self.config

class BioOncoGraphModule(ConfigurableModule):
    """Base module specific to Bio-OncoGraph tasks.
    
    Extends the ConfigurableModule with common properties or methods
    needed across different neural network architectures in the project.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the BioOncoGraphModule.
        
        Args:
            config: A dictionary containing configuration parameters.
        """
        super().__init__(config)
        self.model_name: str = config.get("model_name", "BioOncoGraphBase")
        
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the module. Must be implemented by subclasses.
        
        Args:
            *args: Positional arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.
            
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")
