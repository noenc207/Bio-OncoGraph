"""Metadata structures for tracking multi-modal records and fusions."""

import dataclasses
from typing import Dict


@dataclasses.dataclass(frozen=True)
class ModalityMetadata:
    """Contextual metadata for a single modality record.

    Attributes:
        patient_id: Identifier for the patient.
        modality: Modality name (e.g., 'clinical', 'radiology').
        timestamp: Creation or acquisition timestamp.
        source: Origin of the data (e.g., 'hospital_A', 'sequencer_1').
        dimension: Dimensionality of the extracted features.
        data_type: Underlying data type (e.g., 'float32').
    """
    patient_id: str
    modality: str
    timestamp: str
    source: str
    dimension: int
    data_type: str


@dataclasses.dataclass(frozen=True)
class FusionMetadata:
    """Contextual metadata for a multi-modal fusion result.

    Attributes:
        fusion_id: Unique identifier for the fusion event.
        patient_id: Identifier for the patient.
        strategies_used: Dictionary detailing strategies (e.g., alignment, fusion).
        timestamp: Creation timestamp for the fused representation.
    """
    fusion_id: str
    patient_id: str
    strategies_used: Dict[str, str]
    timestamp: str
