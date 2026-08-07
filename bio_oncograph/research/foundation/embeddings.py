"""Embedding management and evaluation."""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingStore:
    """Stores embeddings and associated metadata."""

    def __init__(self):
        self._embeddings: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []

    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add a single embedding and its metadata."""
        self._embeddings.append(embedding)
        self._metadata.append(metadata)

    def get(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get embedding and metadata at index."""
        return self._embeddings[index], self._metadata[index]

    def get_all(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Stack all embeddings and return with metadata."""
        if not self._embeddings:
            return np.array([]), []
        return np.stack(self._embeddings), self._metadata

    def size(self) -> int:
        """Return the number of stored embeddings."""
        return len(self._embeddings)

    def dimension(self) -> int:
        """Return the dimensionality of the embeddings."""
        if not self._embeddings:
            return 0
        return self._embeddings[0].shape[0]

    def clear(self) -> None:
        """Clear the store."""
        self._embeddings.clear()
        self._metadata.clear()


class EmbeddingEvaluator:
    """Evaluates the quality of graph embeddings."""

    @staticmethod
    def linear_probe_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Simple linear separability score.
        Computes the ratio of inter-class to intra-class variance.
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        global_mean = np.mean(embeddings, axis=0)
        intra_var = 0.0
        inter_var = 0.0
        n = embeddings.shape[0]

        for label in unique_labels:
            class_embs = embeddings[labels == label]
            class_mean = np.mean(class_embs, axis=0)
            n_c = class_embs.shape[0]

            intra_var += np.sum((class_embs - class_mean) ** 2)
            inter_var += n_c * np.sum((class_mean - global_mean) ** 2)

        intra_var /= n
        inter_var /= (len(unique_labels) - 1)

        if intra_var < 1e-8:
            return float('inf')

        return float(inter_var / intra_var)

    @staticmethod
    def silhouette_approximation(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Simplified silhouette score using pairwise distances."""
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
        
        n = min(embeddings.shape[0], 1000)
        idx = np.random.choice(embeddings.shape[0], n, replace=False)
        sub_embs = embeddings[idx]
        sub_labels = labels[idx]

        score = 0.0
        for i in range(n):
            same_class = (sub_labels == sub_labels[i])
            
            if np.sum(same_class) <= 1:
                continue
                
            dist = np.linalg.norm(sub_embs - sub_embs[i], axis=1)
            a = np.mean(dist[same_class])
            
            b_min = float('inf')
            for label in unique_labels:
                if label == sub_labels[i]:
                    continue
                label_mask = (sub_labels == label)
                if np.sum(label_mask) > 0:
                    b = np.mean(dist[label_mask])
                    if b < b_min:
                        b_min = b
                        
            s = (b_min - a) / max(a, b_min)
            score += s
            
        return float(score / n)

    @staticmethod
    def embedding_statistics(embeddings: np.ndarray) -> Dict[str, float]:
        """Compute basic statistics of embeddings."""
        norms = np.linalg.norm(embeddings, axis=1)
        
        cov = np.cov(embeddings, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        isotropy = float(np.max(eigenvalues) / (np.mean(eigenvalues) + 1e-8))

        return {
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms)),
            'isotropy': isotropy
        }
