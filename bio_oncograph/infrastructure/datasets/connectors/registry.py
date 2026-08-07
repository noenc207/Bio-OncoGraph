"""Registry for dataset connectors."""

from typing import Any, Callable, Dict, List, Type

from bio_oncograph.infrastructure.datasets.connectors.base_connector import DatasetConnector


class ConnectorRegistry:
    """Registry for discovering and instantiating specific connectors."""

    _registry: Dict[str, Type[DatasetConnector]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[DatasetConnector]], Type[DatasetConnector]]:
        """Registers a dataset connector class.

        Args:
            name: The name to register the connector under.

        Returns:
            A decorator function.
        """
        def decorator(connector_class: Type[DatasetConnector]) -> Type[DatasetConnector]:
            cls._registry[name] = connector_class
            return connector_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[DatasetConnector]:
        """Retrieves a dataset connector class by name.

        Args:
            name: The name of the registered connector.

        Returns:
            The connector class.

        Raises:
            KeyError: If the connector is not registered.
        """
        if name not in cls._registry:
            raise KeyError(f"Connector '{name}' not found in registry.")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """Lists all available registered connectors.

        Returns:
            A list of registered connector names.
        """
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> DatasetConnector:
        """Creates an instance of a registered dataset connector.

        Args:
            name: The name of the registered connector.
            **kwargs: Keyword arguments to pass to the connector constructor.

        Returns:
            An instance of the connector.
        """
        connector_class = cls.get(name)
        return connector_class(**kwargs)
