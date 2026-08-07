import logging
from abc import ABC, abstractmethod
from typing import Dict, Type

logger = logging.getLogger(__name__)

class ClinicalEndpoint(ABC):
    """Abstract base class for clinical endpoints."""

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the clinical endpoint."""
        pass

    @abstractmethod
    def validate_event(self, time: float, event: int) -> bool:
        """Validates a survival time and event indicator."""
        pass

    @abstractmethod
    def description(self) -> str:
        """Returns a description of the clinical endpoint."""
        pass


class OverallSurvival(ClinicalEndpoint):
    """Overall Survival (OS) clinical endpoint."""

    def name(self) -> str:
        return "OS"

    def validate_event(self, time: float, event: int) -> bool:
        return time >= 0.0 and event in (0, 1)

    def description(self) -> str:
        return "Time from randomization until death from any cause."


class DiseaseFreeSurvival(ClinicalEndpoint):
    """Disease-Free Survival (DFS) clinical endpoint."""

    def name(self) -> str:
        return "DFS"

    def validate_event(self, time: float, event: int) -> bool:
        return time >= 0.0 and event in (0, 1)

    def description(self) -> str:
        return "Time from randomization until disease recurrence or death."


class ProgressionFreeSurvival(ClinicalEndpoint):
    """Progression-Free Survival (PFS) clinical endpoint."""

    def name(self) -> str:
        return "PFS"

    def validate_event(self, time: float, event: int) -> bool:
        return time >= 0.0 and event in (0, 1)

    def description(self) -> str:
        return "Time from randomization until objective tumor progression or death."


class RecurrenceFreeSurvival(ClinicalEndpoint):
    """Recurrence-Free Survival (RFS) clinical endpoint."""

    def name(self) -> str:
        return "RFS"

    def validate_event(self, time: float, event: int) -> bool:
        return time >= 0.0 and event in (0, 1)

    def description(self) -> str:
        return "Time from randomization until objective tumor recurrence."


class EndpointRegistry:
    """Registry for clinical endpoints."""

    _endpoints: Dict[str, Type[ClinicalEndpoint]] = {
        "OS": OverallSurvival,
        "DFS": DiseaseFreeSurvival,
        "PFS": ProgressionFreeSurvival,
        "RFS": RecurrenceFreeSurvival,
    }

    @classmethod
    def register(cls, name: str, endpoint_cls: Type[ClinicalEndpoint]) -> None:
        """Registers a new endpoint class."""
        cls._endpoints[name] = endpoint_cls
        logger.info("Registered clinical endpoint: %s", name)

    @classmethod
    def get_endpoint(cls, name: str) -> ClinicalEndpoint:
        """Retrieves an instantiated endpoint by name."""
        if name not in cls._endpoints:
            raise ValueError(f"Endpoint '{name}' not found in registry.")
        return cls._endpoints[name]()
