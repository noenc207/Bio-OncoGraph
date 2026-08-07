"""
Enumerations for Bio-OncoGraph core states and types.
"""

from enum import Enum, auto


class RunMode(Enum):
    """Execution modes for the Bio-OncoGraph framework."""
    TRAIN = auto()
    EVAL = auto()
    INFERENCE = auto()
    EXPLAIN = auto()


class RelationType(Enum):
    """Types of biological/morphological relationships in the graph."""
    SPATIAL_HOMOPHILY = auto()
    TUMOR_IMMUNE = auto()
    TUMOR_STROMA = auto()
    GENERIC_HETEROPHILY = auto()


class DeviceType(Enum):
    """Hardware device identifiers."""
    CPU = auto()
    CUDA = auto()
    MPS = auto()
