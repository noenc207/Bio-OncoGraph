"""Optimizer factory module for creating parameter groups and optimizers."""

from typing import Any, Dict, Optional

import torch
from torch.optim import Adam, AdamW, SGD


class OptimizerFactory:
    """Factory class for creating PyTorch optimizers with custom parameter groups."""

    @staticmethod
    def create(
        model: torch.nn.Module,
        optimizer_type: str,
        learning_rate: float,
        weight_decay: float = 0.0,
        no_decay_bias_norm: bool = True,
        optimizer_params: Optional[Dict[str, Any]] = None,
    ) -> torch.optim.Optimizer:
        """Creates an optimizer with parameter groups handling (e.g., no weight decay on bias/norm).

        Args:
            model: The neural network model.
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd').
            learning_rate: The base learning rate.
            weight_decay: Weight decay applied to parameters.
            no_decay_bias_norm: If True, disables weight decay for biases and norm layers.
            optimizer_params: Additional parameters for the optimizer.

        Returns:
            The instantiated optimizer.

        Raises:
            ValueError: If an unsupported optimizer type is specified.
        """
        if optimizer_params is None:
            optimizer_params = {}

        if no_decay_bias_norm:
            decay_params = []
            no_decay_params = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
                    
            param_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
        else:
            param_groups = [
                {"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}
            ]

        optimizer_type = optimizer_type.lower()
        if optimizer_type == "adam":
            return Adam(param_groups, lr=learning_rate, **optimizer_params)
        elif optimizer_type == "adamw":
            return AdamW(param_groups, lr=learning_rate, **optimizer_params)
        elif optimizer_type == "sgd":
            return SGD(param_groups, lr=learning_rate, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
