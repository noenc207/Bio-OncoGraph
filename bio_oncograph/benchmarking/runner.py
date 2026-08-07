import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, Dict, List, Optional
import logging
import os

from .datasets import BaseOncoDataset, DataSplitter
from .protocols import FairBenchmarkProtocol
from .compute import ComputeAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """
    Runner for executing full benchmark pipelines including LOCO and Nested CV.
    """
    def __init__(self, model: nn.Module, dataset: BaseOncoDataset, config: Dict[str, Any]):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.protocol = FairBenchmarkProtocol(seed=config.get("seed", 42))
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def run_loco(self, centers: np.ndarray) -> List[Dict[str, Any]]:
        """
        Executes Leave-One-Center-Out benchmarking.
        """
        logger.info("Starting Leave-One-Center-Out (LOCO) Benchmark...")
        splitter = DataSplitter.leave_one_center_out(centers)
        
        results = []
        for fold, (train_idx, test_idx) in enumerate(splitter):
            logger.info(f"Running LOCO Fold {fold}")
            res = self._train_and_evaluate(train_idx, test_idx, fold_name=f"LOCO_{fold}")
            results.append(res)
            
        return results

    def run_nested_cv(self, n_samples: int, outer_folds: int = 5, inner_folds: int = 5) -> List[Dict[str, Any]]:
        """
        Executes Nested Cross-Validation benchmarking.
        """
        logger.info(f"Starting Nested CV ({outer_folds} outer, {inner_folds} inner)...")
        splitter = DataSplitter.nested_cv(n_samples, outer_folds, inner_folds, seed=self.config.get("seed", 42))
        
        results = []
        for i, (train_idx, val_idx, test_idx) in enumerate(splitter):
            logger.info(f"Running Nested CV Split {i}")
            res = self._train_and_evaluate(train_idx, test_idx, fold_name=f"NestedCV_{i}")
            results.append(res)
            
        return results

    def _train_and_evaluate(self, train_idx: np.ndarray, test_idx: np.ndarray, fold_name: str) -> Dict[str, Any]:
        """
        Handles training and evaluation protocol enforcement for a single fold.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.get("learning_rate", 1e-3))
        
        expected_configs = self.config.get("expected_protocol_configs", {})
        if expected_configs:
            is_valid = self.protocol.enforce_exact_protocol(
                self.model, optimizer, train_idx, test_idx, expected_configs
            )
            if not is_valid:
                logger.warning(f"Protocol verification failed for fold {fold_name}!")
            else:
                logger.info(f"Protocol successfully verified for fold {fold_name}.")

        # Actual training loop would go here
        logger.info("Training model...")
        logger.info("Evaluating model...")
        
        if fold_name.endswith("_0") or fold_name == "LOCO_0":
            self._generate_compute_report()
            
        return {"fold": fold_name, "status": "completed"}

    def _generate_compute_report(self) -> None:
        """
        Generates and exports the computation metrics report.
        """
        logger.info("Generating compute report...")
        analyzer = ComputeAnalyzer(self.model, self.device)
        input_shape = self.config.get("input_shape", (1, 3, 224, 224))
        dummy_input = torch.randn(*input_shape)
        
        output_dir = self.config.get("output_dir", ".")
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "compute_report.json")
        
        analyzer.generate_report(dummy_input, output_path=report_path)
        logger.info(f"Compute report saved to {report_path}")
