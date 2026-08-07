"""Bio-OncoGraph Main Model Module."""
import numpy as np
from typing import Dict, Any, Optional

from bio_oncograph.ai.model.encoder import GraphEncoder
from bio_oncograph.ai.model.pooling import GraphPooling
from bio_oncograph.ai.model.fusion import FeatureFusion
from bio_oncograph.ai.model.heads import ClassificationHead, SurvivalHead


class BioOncoGraphOutput:
    """Output container for Bio-OncoGraph model."""
    
    def __init__(self, logits: np.ndarray, risk_score: float) -> None:
        """Initialize the output container.

        Args:
            logits: Classification logits.
            risk_score: Survival risk score.
        """
        self.logits = logits
        self.risk_score = risk_score


class BioOncoGraphModel:
    """Bio-OncoGraph integrative model architecture.

    Attributes:
        config (Dict[str, Any]): Model configuration.
        encoder (GraphEncoder): Graph representation learning module.
        pooling (GraphPooling): Node embedding aggregation module.
        fusion (FeatureFusion): Multi-modal feature fusion module.
        classification_head (ClassificationHead): Logits predictor.
        survival_head (SurvivalHead): Risk score predictor.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the model components.

        Args:
            config: Full model configuration dictionary.
        """
        self.config = config
        
        encoder_config = config.get("encoder", {})
        pooling_config = config.get("pooling", {})
        fusion_config = config.get("fusion", {})
        heads_config = config.get("heads", {})
        
        self.encoder = GraphEncoder(encoder_config)
        self.pooling = GraphPooling(pooling_config)
        self.fusion = FeatureFusion(fusion_config)
        self.classification_head = ClassificationHead(heads_config.get("classification", {}))
        self.survival_head = SurvivalHead(heads_config.get("survival", {}))

    def encode(self, graph: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Encode the input graph.

        Args:
            graph: Graph dictionary containing nodes and edges.

        Returns:
            Dictionary of node embeddings.
        """
        return self.encoder.forward(graph)

    def pool(self, node_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Pool node embeddings into a graph embedding.

        Args:
            node_embeddings: Output from the encoder.

        Returns:
            Aggregated graph embedding.
        """
        return self.pooling.forward(node_embeddings)

    def fuse(self, features: np.ndarray, aux: np.ndarray) -> np.ndarray:
        """Fuse graph embedding with auxiliary features.

        Args:
            features: Graph embedding.
            aux: Auxiliary clinical features.

        Returns:
            Fused multimodal embedding.
        """
        return self.fusion.forward(features, aux)

    def predict(self, features: np.ndarray) -> BioOncoGraphOutput:
        """Generate predictions from fused features.

        Args:
            features: Fused features array.

        Returns:
            BioOncoGraphOutput containing logits and risk score.
        """
        logits = self.classification_head.forward(features)
        risk = self.survival_head.forward(features)
        return BioOncoGraphOutput(logits=logits, risk_score=risk)

    def validate(self, graph: Dict[str, Any]) -> bool:
        """Validate input graph structure.

        Args:
            graph: Input graph dictionary.

        Returns:
            True if valid, False otherwise.
        """
        if "nodes" not in graph or "edges" not in graph:
            return False
        return True

    def forward(self, graph: Dict[str, Any], aux_features: np.ndarray) -> Optional[BioOncoGraphOutput]:
        """Execute the full forward pass pipeline.

        Args:
            graph: Graph data dictionary.
            aux_features: Auxiliary features array.

        Returns:
            BioOncoGraphOutput containing predictions, or None if validation fails.
        """
        if not self.validate(graph):
            return None
            
        node_embeddings = self.encode(graph)
        graph_embedding = self.pool(node_embeddings)
        fused_features = self.fuse(graph_embedding, aux_features)
        return self.predict(fused_features)

    def report(self) -> str:
        """Generate a textual report of the model architecture.

        Returns:
            Architecture summary string.
        """
        summary = [
            "Bio-OncoGraph M015 Architecture:",
            f"Encoder: Multi-layer GCN",
            f"Pooling: {self.pooling.pooling_type}",
            "Fusion: Linear Concatenation Projection",
            "Heads: Classification (Logits) and Survival (Risk Score)"
        ]
        return "\n".join(summary)
