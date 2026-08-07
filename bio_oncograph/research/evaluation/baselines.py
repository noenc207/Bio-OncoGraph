import abc
import logging
import numpy as np

logger = logging.getLogger(__name__)

class BaselineModel(abc.ABC):
    """Abstract base class for baseline models."""
    
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the model."""
        pass

class ClassicalMLBaseline(BaselineModel):
    """Mock Random Forest/SVM baseline."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Fitting ClassicalMLBaseline")

    def predict(self, X: np.ndarray) -> np.ndarray:
        logger.info("Predicting with ClassicalMLBaseline")
        return np.zeros(X.shape[0])

class DeepLearningBaseline(BaselineModel):
    """Mock MLP/CNN baseline."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Fitting DeepLearningBaseline")

    def predict(self, X: np.ndarray) -> np.ndarray:
        logger.info("Predicting with DeepLearningBaseline")
        return np.zeros(X.shape[0])

class GraphLearningBaseline(BaselineModel):
    """Mock GNN baseline."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Fitting GraphLearningBaseline")

    def predict(self, X: np.ndarray) -> np.ndarray:
        logger.info("Predicting with GraphLearningBaseline")
        return np.zeros(X.shape[0])

class MultimodalBaseline(BaselineModel):
    """Mock Multimodal fusion baseline."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Fitting MultimodalBaseline")

    def predict(self, X: np.ndarray) -> np.ndarray:
        logger.info("Predicting with MultimodalBaseline")
        return np.zeros(X.shape[0])
