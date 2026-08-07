"""Loss reporting."""

import dataclasses
from typing import Tuple, Dict, Any

@dataclasses.dataclass(frozen=True)
class LossReport:
    """Report of a loss validation.
    
    Attributes:
        is_valid: Whether the loss state is valid.
        warnings: Tuple of warning messages.
        metadata: Additional metadata.
    """
    is_valid: bool
    warnings: Tuple[str, ...]
    metadata: Dict[str, Any]
