"""Validator for determinism and reproducibility."""

from typing import Any, Callable
import numpy as np


class ReproducibilityValidator:
    """Validates the reproducibility and determinism of a pipeline."""

    def _compare_outputs(self, out1: Any, out2: Any) -> bool:
        """Compare two outputs for equality.
        
        Args:
            out1: First output.
            out2: Second output.
            
        Returns:
            True if outputs are equal, False otherwise.
        """
        if isinstance(out1, np.ndarray) and isinstance(out2, np.ndarray):
            return bool(np.array_equal(out1, out2))
        elif isinstance(out1, dict) and isinstance(out2, dict):
            if set(out1.keys()) != set(out2.keys()):
                return False
            return all(self._compare_outputs(out1[k], out2[k]) for k in out1)
        elif isinstance(out1, (list, tuple)) and isinstance(out2, (list, tuple)):
            if len(out1) != len(out2):
                return False
            return all(self._compare_outputs(i1, i2) for i1, i2 in zip(out1, out2))
        else:
            return out1 == out2

    def check_determinism(self, pipeline_callable: Callable[[Any], Any], inputs: Any, num_trials: int = 3) -> bool:
        """Check if a pipeline produces deterministic outputs.
        
        Args:
            pipeline_callable: The pipeline function to test.
            inputs: Inputs to provide to the pipeline.
            num_trials: Number of trials to run.
            
        Returns:
            True if all runs produce the same output, False otherwise.
        """
        if num_trials < 2:
            return True
            
        base_output = pipeline_callable(inputs)
        
        for _ in range(num_trials - 1):
            current_output = pipeline_callable(inputs)
            if not self._compare_outputs(base_output, current_output):
                return False
                
        return True
