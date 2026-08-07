import dataclasses
import logging
from typing import Any, Dict, List, Type

logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class ExtensionPoint:
    """Defines an integration point for extensions in the open science platform."""
    name: str
    interface_type: Type
    description: str

class ExtensionRegistry:
    """Registry for managing extension points and their implementations."""
    
    def __init__(self) -> None:
        self._points: Dict[str, ExtensionPoint] = {}
        self._extensions: Dict[str, List[Any]] = {}
        logger.info("Initialized ExtensionRegistry.")
        
    def register_extension_point(self, point: ExtensionPoint) -> None:
        """Register a new extension point.
        
        Args:
            point: The ExtensionPoint definition.
            
        Raises:
            ValueError: If point name is already registered.
        """
        if point.name in self._points:
            error_msg = f"Extension point '{point.name}' already registered."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        self._points[point.name] = point
        self._extensions[point.name] = []
        logger.info(f"Registered extension point: {point.name}")
        
    def register_extension(self, point_name: str, implementation: Any) -> None:
        """Register an implementation for an extension point.
        
        Args:
            point_name: The name of the extension point.
            implementation: The extension instance.
            
        Raises:
            KeyError: If extension point does not exist.
            TypeError: If implementation does not match interface type.
        """
        if point_name not in self._points:
            error_msg = f"Extension point '{point_name}' not found."
            logger.error(error_msg)
            raise KeyError(error_msg)
            
        point = self._points[point_name]
        if not isinstance(implementation, point.interface_type):
            error_msg = f"Implementation must be of type {point.interface_type.__name__}."
            logger.error(error_msg)
            raise TypeError(error_msg)
            
        self._extensions[point_name].append(implementation)
        logger.info(f"Registered extension for point: {point_name}")
        
    def get_extensions(self, point_name: str) -> List[Any]:
        """Get all extensions registered for a point.
        
        Args:
            point_name: The name of the extension point.
            
        Returns:
            A list of extension implementations.
            
        Raises:
            KeyError: If extension point does not exist.
        """
        if point_name not in self._extensions:
            error_msg = f"Extension point '{point_name}' not found."
            logger.error(error_msg)
            raise KeyError(error_msg)
            
        return list(self._extensions[point_name])
        
    def list_extension_points(self) -> List[ExtensionPoint]:
        """List all registered extension points.
        
        Returns:
            A list of ExtensionPoint definitions.
        """
        return list(self._points.values())
