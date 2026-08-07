"""Output definitions for the Bio-OncoGraph M015 Model."""

import dataclasses
from typing import Dict, Optional, Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class BioOncoGraphOutput:
    """Output structure of the Bio-OncoGraph model forward pass.
    
    Attributes:
        node_embeddings: A dictionary mapping node IDs to their embeddings.
        graph_embedding: A single vector representing the entire graph.
        logits: Output logits for classification tasks, if enabled.
        survival_prediction: Output predictions for survival tasks, if enabled.
        auxiliary_outputs: Any additional outputs generated during the forward pass.
        metadata: Any relevant metadata associated with the output.
    """
    node_embeddings: Dict[Any, np.ndarray]
    graph_embedding: np.ndarray
    logits: Optional[np.ndarray]
    survival_prediction: Optional[np.ndarray]
    auxiliary_outputs: Dict[str, Any]
    metadata: Dict[str, Any]
