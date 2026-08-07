"""Metrics tracking."""
import logging
from typing import Dict, Optional

import torch
import torchmetrics

logger = logging.getLogger(__name__)


class MetricTracker:
    """Tracks and computes various evaluation metrics.
    
    Includes Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
    """

    def __init__(self, task: str = "binary", num_classes: Optional[int] = None) -> None:
        """Initializes the MetricTracker.

        Args:
            task: Type of task, e.g., 'binary', 'multiclass', 'multilabel'.
            num_classes: Number of classes, required if task is 'multiclass'.
        """
        self.task = task
        self.num_classes = num_classes
        
        kwargs = {"task": task}
        if task in ("multiclass", "multilabel") and num_classes is not None:
            kwargs["num_classes"] = num_classes

        self.accuracy = torchmetrics.Accuracy(**kwargs)
        self.precision = torchmetrics.Precision(**kwargs)
        self.recall = torchmetrics.Recall(**kwargs)
        self.f1 = torchmetrics.F1Score(**kwargs)
        self.roc_auc = torchmetrics.AUROC(**kwargs)
        self.pr_auc = torchmetrics.AveragePrecision(**kwargs)
        self.calibration_error = torchmetrics.CalibrationError(**kwargs)
        
        self.metrics = torch.nn.ModuleDict({
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "calibration_error": self.calibration_error,
        })
        self._device: Optional[torch.device] = None

    def _ensure_device(self, device: torch.device) -> None:
        """Moves metrics to the correct device."""
        if self._device != device:
            self.metrics.to(device)
            self._device = device

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Updates the state of all tracked metrics.

        Args:
            preds: Model predictions.
            target: Ground truth targets.
        """
        self._ensure_device(preds.device)
        
        if preds.is_floating_point() and not torch.all((preds >= 0) & (preds <= 1)):
            preds_prob = torch.sigmoid(preds) if self.task == "binary" else torch.softmax(preds, dim=-1)
        else:
            preds_prob = preds

        for name, metric in self.metrics.items():
            try:
                if name in ("roc_auc", "pr_auc", "calibration_error"):
                    metric.update(preds_prob, target)
                else:
                    metric.update(preds, target)
            except Exception as e:
                logger.debug(f"Failed to update metric {name}: {e}")

    def compute(self) -> Dict[str, float]:
        """Computes the final metric values.

        Returns:
            Dictionary mapping metric names to their computed values.
        """
        results = {}
        for name, metric in self.metrics.items():
            try:
                val = metric.compute()
                results[name] = val.item() if isinstance(val, torch.Tensor) else val
            except Exception as e:
                logger.debug(f"Failed to compute metric {name}: {e}")
                results[name] = float("nan")
        return results

    def reset(self) -> None:
        """Resets the state of all metrics."""
        for metric in self.metrics.values():
            metric.reset()
