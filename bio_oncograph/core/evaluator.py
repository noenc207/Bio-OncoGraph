"""
Evaluator for Bio-OncoGraph.

Provides ``Evaluator`` — an orchestrator that drives
``MetricCollection`` through the batch / epoch / summary lifecycle.
The Evaluator is the canonical integration point between the
``Trainer`` (M008) and the Metrics Framework (M010).

All types are pure Python — **no** ``torch`` imports.

Design references:
  - Software Architecture § 14 (Metrics)
  - Software Architecture § 19 (Training Engine)
"""

import logging
import math
import csv
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

from bio_oncograph.core.metrics import MetricCollection

logger = logging.getLogger("BioOncoGraph")


class Evaluator:
    """
    Metric evaluation orchestrator.

    Lifecycle::

        evaluate_batch() × B  →  evaluate_epoch()  →  repeat …
        summary()  →  reset()

    The Evaluator drives a ``MetricCollection`` and stores
    per-epoch results for post-hoc analysis.

    Args:
        metrics: The metric collection to evaluate.
    """

    def __init__(self, metrics: MetricCollection) -> None:
        """
        Initialise the evaluator.

        Args:
            metrics: A pre-configured ``MetricCollection``.
        """
        self._metrics = metrics
        self._epoch_results: List[Dict[str, float]] = []

    @property
    def metrics(self) -> MetricCollection:
        """Return the underlying metric collection."""
        return self._metrics

    @property
    def num_epochs(self) -> int:
        """Return the number of completed epochs."""
        return len(self._epoch_results)

    def evaluate_batch(
        self, prediction: Any, target: Any
    ) -> None:
        """
        Forward one batch to all metrics.

        Args:
            prediction: Predicted value(s).
            target: Ground-truth value(s).
        """
        self._metrics.update(prediction, target)

    def evaluate_epoch(self) -> Dict[str, float]:
        """
        Compute metrics for the current epoch, store the result,
        and reset all metrics for the next epoch.

        Returns:
            Dictionary mapping metric name → value.
        """
        results = self._metrics.compute()
        self._epoch_results.append(dict(results))
        self._metrics.reset()
        logger.info(
            "Epoch %d metrics: %s",
            len(self._epoch_results) - 1,
            results,
        )
        return results

    def summary(self) -> Dict[str, Any]:
        """
        Return a summary of all evaluated epochs.

        Returns:
            Dictionary with ``num_epochs``, ``epochs`` (list of
            per-epoch dicts), and ``latest`` (last epoch's metrics
            or empty dict).
        """
        return {
            "num_epochs": len(self._epoch_results),
            "epochs": list(self._epoch_results),
            "latest": (
                self._epoch_results[-1]
                if self._epoch_results
                else {}
            ),
        }

    def reset(self) -> None:
        """Reset metrics and discard all epoch results."""
        self._metrics.reset()
        self._epoch_results.clear()
        logger.info("Evaluator reset")

class MultiSeedAggregator:
    """Aggregates metric results across multiple random seeds."""
    
    def __init__(self, target_seeds: List[int] = None) -> None:
        self.target_seeds = target_seeds or [42, 123, 777, 2026, 3407]
        self.results_by_seed: Dict[int, Dict[str, float]] = {}

    def add_result(self, seed: int, metrics: Dict[str, float]) -> None:
        self.results_by_seed[seed] = metrics

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        if not self.results_by_seed:
            return {}
        
        aggregated = {}
        metric_keys = list(self.results_by_seed.values())[0].keys()
        
        for key in metric_keys:
            values = [res[key] for res in self.results_by_seed.values() if key in res]
            n = len(values)
            if n == 0:
                continue
            mean = sum(values) / n
            if n > 1:
                variance = sum((x - mean) ** 2 for x in values) / (n - 1)
                std = math.sqrt(variance)
            else:
                std = 0.0
            
            # 95% CI is approx mean +- 1.96 * std / sqrt(n)
            ci_95 = 1.96 * std / math.sqrt(n) if n > 0 else 0.0
            
            aggregated[key] = {
                "mean": mean,
                "std": std,
                "ci_95": ci_95
            }
        return aggregated


class PredictionArtifactGenerator:
    """Generates artifacts from predictions, such as CSVs and Numpy files."""
    
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "attention_maps").mkdir(parents=True, exist_ok=True)
        
    def generate(self, predictions: List[Dict[str, Any]]) -> None:
        """
        Generate artifacts from a list of prediction dictionaries.
        
        Each prediction dict may contain:
        'id', 'prediction', 'probability', 'embedding', 'trust_score', 'attention_map'.
        """
        if not predictions:
            return
            
        preds_path = self.output_dir / "predictions.csv"
        probs_path = self.output_dir / "probabilities.csv"
        trust_path = self.output_dir / "trust_scores.csv"
        
        embeddings = []
        
        with open(preds_path, "w", newline="", encoding="utf-8") as f_pred, \
             open(probs_path, "w", newline="", encoding="utf-8") as f_prob, \
             open(trust_path, "w", newline="", encoding="utf-8") as f_trust:
             
             pred_writer = csv.writer(f_pred)
             prob_writer = csv.writer(f_prob)
             trust_writer = csv.writer(f_trust)
             
             pred_writer.writerow(["id", "prediction"])
             prob_writer.writerow(["id", "probability"])
             trust_writer.writerow(["id", "trust_score"])
             
             for idx, p in enumerate(predictions):
                 sample_id = p.get("id", str(idx))
                 pred_writer.writerow([sample_id, p.get("prediction", 0)])
                 prob_writer.writerow([sample_id, p.get("probability", 0.0)])
                 trust_writer.writerow([sample_id, p.get("trust_score", 0.0)])
                 
                 emb = p.get("embedding")
                 if emb is not None:
                     embeddings.append(emb)
                 
                 attn = p.get("attention_map")
                 if attn is not None:
                     np.save(self.output_dir / "attention_maps" / f"{sample_id}_attn.npy", attn)
                     
        if embeddings:
            np.save(self.output_dir / "embeddings.npy", np.stack(embeddings))
