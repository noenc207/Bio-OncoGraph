"""Abstract trainer interface."""
import abc
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer(abc.ABC):
    """Abstract base class for trainers.
    
    This class defines the minimal interface required for training
    and evaluating a model.
    """

    @abc.abstractmethod
    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """Runs a single training epoch.

        Args:
            model: The neural network model to train.
            dataloader: DataLoader providing the training data.
            optimizer: The optimization algorithm.
            criterion: The loss function.
            device: The device to run training on (CPU/GPU).
            scaler: Optional gradient scaler for mixed precision.

        Returns:
            A dictionary containing training metrics for the epoch.
        """
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluates the model on a validation or test dataset.

        Args:
            model: The neural network model to evaluate.
            dataloader: DataLoader providing the evaluation data.
            criterion: The loss function.
            device: The device to run evaluation on.

        Returns:
            A dictionary containing evaluation metrics.
        """
        pass
