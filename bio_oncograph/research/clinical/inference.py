"""Inference utilities for clinical models."""
import logging
from typing import List
import numpy as np

from .engine import ClinicalOutcomeEngine, ClinicalPredictionOutput

logger = logging.getLogger(__name__)


class ClinicalInference:
    """Inference interface for clinical models."""

    @staticmethod
    def predict_batch(engine: ClinicalOutcomeEngine, features_batch: List[np.ndarray]) -> List[ClinicalPredictionOutput]:
        """Generate predictions for a batch of features.
        
        Args:
            engine: The clinical outcome engine to use for prediction.
            features_batch: List of feature arrays.
            
        Returns:
            List of ClinicalPredictionOutput objects.
        """
        logger.info("Running batch prediction for %d feature sets.", len(features_batch))
        
        # Ensure determinism by temporarily setting the global seed if the engine has a seed configured
        seed = engine.config.seed
        if seed is not None:
            np.random.seed(seed)
            
        outputs = []
        for features in features_batch:
            output = engine.predict(features)
            outputs.append(output)
            
        logger.info("Batch prediction complete.")
        return outputs

    @staticmethod
    def predict_single(engine: ClinicalOutcomeEngine, features: np.ndarray) -> ClinicalPredictionOutput:
        """Generate a prediction for a single set of features.
        
        Args:
            engine: The clinical outcome engine to use for prediction.
            features: Feature array for a single cohort or patient group.
            
        Returns:
            A ClinicalPredictionOutput object.
        """
        logger.info("Running single prediction.")
        
        # Ensure determinism
        seed = engine.config.seed
        if seed is not None:
            np.random.seed(seed)
            
        output = engine.predict(features)
        logger.info("Single prediction complete.")
        return output
