import abc
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class PluginInterface(abc.ABC):
    """Abstract base class for open science plugins."""
    
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the plugin."""
        pass
        
    @abc.abstractmethod
    def version(self) -> str:
        """Return the version of the plugin."""
        pass
        
    @abc.abstractmethod
    def initialize(self, config: Dict) -> None:
        """Initialize the plugin with the given configuration."""
        pass
        
    @abc.abstractmethod
    def execute(self, inputs: Dict) -> Dict:
        """Execute the plugin functionality."""
        pass

class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self) -> None:
        """Initialize an empty plugin registry."""
        self._plugins: Dict[str, PluginInterface] = {}
        logger.info("Initialized PluginRegistry.")
        
    def register_plugin(self, plugin: PluginInterface) -> None:
        """Register a plugin.
        
        Args:
            plugin: The plugin instance to register.
            
        Raises:
            ValueError: If a plugin with the same name is already registered.
        """
        name = plugin.name()
        if name in self._plugins:
            error_msg = f"Plugin '{name}' is already registered."
            logger.error(error_msg)
            raise ValueError(error_msg)
        self._plugins[name] = plugin
        logger.info(f"Registered plugin: {name} (version {plugin.version()})")
        
    def get_plugin(self, name: str) -> PluginInterface:
        """Get a plugin by name.
        
        Args:
            name: The name of the plugin.
            
        Returns:
            The requested plugin instance.
            
        Raises:
            KeyError: If the plugin is not found.
        """
        if name not in self._plugins:
            error_msg = f"Plugin '{name}' not found in registry."
            logger.error(error_msg)
            raise KeyError(error_msg)
        return self._plugins[name]
        
    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self._plugins.keys())
        
    def unregister_plugin(self, name: str) -> None:
        """Unregister a plugin by name.
        
        Args:
            name: The name of the plugin to remove.
        """
        if name in self._plugins:
            del self._plugins[name]
            logger.info(f"Unregistered plugin: {name}")
        else:
            logger.warning(f"Attempted to unregister non-existent plugin: {name}")
            
    def validate_plugin(self, plugin: PluginInterface) -> List[str]:
        """Validate a plugin instance.
        
        Args:
            plugin: The plugin to validate.
            
        Returns:
            A list of validation error messages. Empty if valid.
        """
        errors: List[str] = []
        if not isinstance(plugin, PluginInterface):
            errors.append("Plugin must inherit from PluginInterface.")
        try:
            name = plugin.name()
            if not isinstance(name, str) or not name.strip():
                errors.append("Plugin name must be a non-empty string.")
        except Exception as e:
            errors.append(f"Failed to get plugin name: {e}")
            
        try:
            version = plugin.version()
            if not isinstance(version, str) or not version.strip():
                errors.append("Plugin version must be a non-empty string.")
        except Exception as e:
            errors.append(f"Failed to get plugin version: {e}")
            
        return errors
