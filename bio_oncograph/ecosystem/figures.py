"""Scientific figure generation utilities."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FigureSpec:
    """Specification for a scientific figure."""
    figure_type: str
    title: str
    xlabel: str
    ylabel: str
    data: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


class FigureGenerator:
    """Generator for scientific figures and associated data."""

    def generate_roc_data(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute TPR and FPR arrays for ROC curves.
        
        Args:
            y_true: True binary labels.
            y_scores: Target scores/probabilities.
            
        Returns:
            Dictionary containing 'fpr' and 'tpr' arrays.
        """
        thresholds = np.sort(np.unique(y_scores))[::-1]
        tpr = np.zeros_like(thresholds, dtype=float)
        fpr = np.zeros_like(thresholds, dtype=float)
        
        num_pos = np.sum(y_true == 1)
        num_neg = np.sum(y_true == 0)
        
        for i, t in enumerate(thresholds):
            preds = (y_scores >= t).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            
            tpr[i] = tp / num_pos if num_pos > 0 else 0.0
            fpr[i] = fp / num_neg if num_neg > 0 else 0.0
            
        logger.info("Generated ROC data")
        return {"fpr": fpr, "tpr": tpr}

    def generate_pr_data(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute precision and recall arrays for PR curves.
        
        Args:
            y_true: True binary labels.
            y_scores: Target scores/probabilities.
            
        Returns:
            Dictionary containing 'precision' and 'recall' arrays.
        """
        thresholds = np.sort(np.unique(y_scores))[::-1]
        precision = np.zeros_like(thresholds, dtype=float)
        recall = np.zeros_like(thresholds, dtype=float)
        
        num_pos = np.sum(y_true == 1)
        
        for i, t in enumerate(thresholds):
            preds = (y_scores >= t).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall[i] = tp / num_pos if num_pos > 0 else 0.0
            
        logger.info("Generated PR data")
        return {"precision": precision, "recall": recall}

    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> np.ndarray:
        """Generate a confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            labels: List of unique string labels.
            
        Returns:
            A numpy array representing the confusion matrix.
        """
        n_classes = len(labels)
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        for t, p in zip(y_true, y_pred):
            if t in label_to_idx and p in label_to_idx:
                matrix[label_to_idx[t], label_to_idx[p]] += 1
                
        logger.info("Generated confusion matrix")
        return matrix

    def generate_kaplan_meier_data(self, times: np.ndarray, events: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute survival curve points.
        
        Args:
            times: Event or censoring times.
            events: Binary indicators (1=event occurred, 0=censored).
            
        Returns:
            Dictionary containing 'times' and 'survival_probabilities'.
        """
        unique_times = np.sort(np.unique(times))
        survival = np.ones_like(unique_times, dtype=float)
        
        n_at_risk = len(times)
        current_survival = 1.0
        
        for i, t in enumerate(unique_times):
            events_at_t = np.sum((times == t) & (events == 1))
            censored_at_t = np.sum((times == t) & (events == 0))
            
            if n_at_risk > 0:
                current_survival *= (1.0 - events_at_t / n_at_risk)
                
            survival[i] = current_survival
            n_at_risk -= (events_at_t + censored_at_t)
            
        logger.info("Generated Kaplan-Meier data")
        return {"times": unique_times, "survival_probabilities": survival}

    def generate_calibration_data(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> Dict[str, np.ndarray]:
        """Compute calibration data.
        
        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities.
            n_bins: Number of bins.
            
        Returns:
            Dictionary containing 'mean_predicted_value' and 'fraction_of_positives'.
        """
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        indices = np.digitize(y_prob, bins) - 1
        
        bin_sums = np.bincount(indices, weights=y_prob, minlength=n_bins)
        bin_true = np.bincount(indices, weights=y_true, minlength=n_bins)
        bin_total = np.bincount(indices, minlength=n_bins)
        
        nonzero = bin_total > 0
        mean_pred = np.zeros(n_bins)
        frac_pos = np.zeros(n_bins)
        
        mean_pred[nonzero] = bin_sums[nonzero] / bin_total[nonzero]
        frac_pos[nonzero] = bin_true[nonzero] / bin_total[nonzero]
        
        logger.info("Generated calibration data")
        return {
            "mean_predicted_value": mean_pred[nonzero],
            "fraction_of_positives": frac_pos[nonzero]
        }

    def generate_ablation_table(self, results: Dict[str, Dict[str, float]]) -> str:
        """Generate a markdown table for ablation study results.
        
        Args:
            results: Dictionary mapping configuration name to a dictionary of metrics.
            
        Returns:
            A string containing the markdown table.
        """
        if not results:
            return ""
            
        configs = list(results.keys())
        metrics = list(results[configs[0]].keys())
        
        lines = ["| Configuration | " + " | ".join(metrics) + " |",
                 "| " + " | ".join(["---"] * (len(metrics) + 1)) + " |"]
                 
        for config in configs:
            row = [config]
            for metric in metrics:
                row.append(f"{results[config].get(metric, 0.0):.4f}")
            lines.append("| " + " | ".join(row) + " |")
            
        logger.info("Generated ablation table")
        return "\n".join(lines)
