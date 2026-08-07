"""Reporting utilities for the Bio-OncoGraph M015 Model."""

import dataclasses
from typing import Tuple, Dict, Any


@dataclasses.dataclass(frozen=True)
class ModelReport:
    """A report detailing the status and metadata of a model or operation.
    
    Attributes:
        is_valid: Whether the model or operation is considered valid.
        warnings: A tuple of warning messages generated.
        metadata: Additional metadata related to the report.
    """
    is_valid: bool
    warnings: Tuple[str, ...]
    metadata: Dict[str, Any]
