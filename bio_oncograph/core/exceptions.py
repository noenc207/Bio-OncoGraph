"""
Exception hierarchy for Bio-OncoGraph framework.
"""


class BioOncoGraphException(Exception):
    """Base exception for all Bio-OncoGraph errors."""
    pass


class BioOncoGraphComputeError(BioOncoGraphException):
    """Errors arising during mathematical or graph computations."""
    pass


class DataException(BioOncoGraphException):
    """Errors related to data loading, parsing, or structural integrity."""
    pass


class InvalidGraphTopologyException(DataException):
    """Raised when a graph has invalid topology (e.g., disconnected components inappropriately)."""
    pass


class MissingClinicalDataException(DataException):
    """Raised when required clinical labels (e.g., survival targets) are missing."""
    pass


class ModelException(BioOncoGraphException):
    """Errors related to neural network module state or forward pass."""
    pass


class NaNGradientException(ModelException):
    """Raised when gradients or intermediate tensors evaluate to NaN."""
    pass


class TopologyCollapseException(ModelException):
    """Raised during pooling if the number of active nodes collapses to 0."""
    pass


class ConfigException(BioOncoGraphException):
    """Errors related to Hydra/OmegaConf configuration validation."""
    pass


class HardwareCapacityError(BioOncoGraphException):
    """Raised when out-of-memory or device capability limits are hit."""
    pass


class ContractViolationError(BioOncoGraphException):
    """Raised when a tensor shape, type, or internal API contract is violated."""
    pass


class ContainerError(BioOncoGraphException):
    """Base exception for Dependency Injection container errors."""
    pass


class DependencyNotFoundError(ContainerError):
    """Raised when a required dependency cannot be resolved."""
    pass


class CircularDependencyError(ContainerError):
    """Raised when a circular dependency chain is detected."""
    pass


class PipelineValidationError(BioOncoGraphException):
    """Raised when the pipeline dependency graph fails validation."""
    pass
