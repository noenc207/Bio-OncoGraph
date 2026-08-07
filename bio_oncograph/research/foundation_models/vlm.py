"""Vision-Language Model abstractions."""

import logging
from typing import Dict, List

import numpy as np

from bio_oncograph.research.foundation_models.encoders import ImageEncoder, TextEncoder
from bio_oncograph.research.foundation_models.adapters import FoundationAdapter


logger = logging.getLogger(__name__)


class VisionLanguageModel:
    """Vision-Language Model."""

    def __init__(
        self,
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        multimodal_adapter: FoundationAdapter
    ) -> None:
        """Initializes the VLM.

        Args:
            image_encoder: Encoder for image modalities.
            text_encoder: Encoder for text modalities.
            multimodal_adapter: Adapter for multimodal fusion.
        """
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.multimodal_adapter = multimodal_adapter
        self.multimodal_adapter.load_weights()

    def generate(self, image_embeddings: np.ndarray, text_prompts: List[str]) -> Dict[str, np.ndarray]:
        """Generates outputs based on images and texts.

        Args:
            image_embeddings: Pre-encoded image representations.
            text_prompts: List of text prompts.

        Returns:
            A dictionary containing generated outputs.
        """
        logger.info("Generating VLM output.")
        text_embeddings = np.random.rand(len(text_prompts), image_embeddings.shape[-1])
        output = self.multimodal_adapter.forward({
            'image_embeds': image_embeddings,
            'text_embeds': text_embeddings
        })
        return {'output': output}

    def score_similarity(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        """Scores similarity between image and text embeddings.

        Args:
            image_embeddings: Image embeddings.
            text_embeddings: Text embeddings.

        Returns:
            Similarity scores as a numpy array.
        """
        logger.info("Scoring image-text similarity.")
        return np.dot(image_embeddings, text_embeddings.T)
