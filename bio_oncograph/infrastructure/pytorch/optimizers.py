import torch
from torch.optim import Adam, AdamW, SGD
from typing import Any, Dict

class OptimizerRegistry:
    @staticmethod
    def get_optimizer(name: str, model_parameters, **kwargs) -> torch.optim.Optimizer:
        name_lower = name.lower()
        if name_lower == "adam":
            return Adam(model_parameters, **kwargs)
        elif name_lower == "adamw":
            return AdamW(model_parameters, **kwargs)
        elif name_lower == "sgd":
            return SGD(model_parameters, **kwargs)
        elif name_lower == "lion":
            try:
                from lion_pytorch import Lion
                return Lion(model_parameters, **kwargs)
            except ImportError:
                raise ImportError("lion_pytorch is required for Lion optimizer")
        elif name_lower == "lamb":
            try:
                import torch_optimizer as optim
                return optim.Lamb(model_parameters, **kwargs)
            except ImportError:
                raise ImportError("torch_optimizer is required for LAMB")
        else:
            raise ValueError(f"Unknown optimizer: {name}")
