import logging
from typing import Callable, Dict, Any

from .plugins import PluginInterface

logger = logging.getLogger(__name__)

class FunctionPluginAdapter(PluginInterface):
    """Adapter to convert a simple function into a PluginInterface."""
    
    def __init__(self, name: str, func: Callable, version_str: str = "1.0.0") -> None:
        self._name = name
        self._func = func
        self._version = version_str
        
    def name(self) -> str:
        return self._name
        
    def version(self) -> str:
        return self._version
        
    def initialize(self, config: Dict) -> None:
        logger.info(f"Initialized plugin {self._name} with config {config}")
        
    def execute(self, inputs: Dict) -> Dict:
        logger.info(f"Executing plugin {self._name}")
        return self._func(inputs)

class PluginSDK:
    """SDK for rapidly creating platform plugins."""
    
    @staticmethod
    def create_dataset_plugin(name: str, loader_fn: Callable) -> PluginInterface:
        """Create a dataset loader plugin.
        
        Args:
            name: Plugin name.
            loader_fn: Function that takes inputs and returns dataset dict.
            
        Returns:
            A PluginInterface implementation.
        """
        logger.info(f"Creating dataset plugin: {name}")
        return FunctionPluginAdapter(name, loader_fn)
        
    @staticmethod
    def create_model_plugin(name: str, model_fn: Callable) -> PluginInterface:
        """Create a model provider plugin.
        
        Args:
            name: Plugin name.
            model_fn: Function that takes inputs and returns model outputs dict.
            
        Returns:
            A PluginInterface implementation.
        """
        logger.info(f"Creating model plugin: {name}")
        return FunctionPluginAdapter(name, model_fn)
        
    @staticmethod
    def create_loss_plugin(name: str, loss_fn: Callable) -> PluginInterface:
        """Create a loss function plugin.
        
        Args:
            name: Plugin name.
            loss_fn: Function that computes loss.
            
        Returns:
            A PluginInterface implementation.
        """
        logger.info(f"Creating loss plugin: {name}")
        return FunctionPluginAdapter(name, loss_fn)
        
    @staticmethod
    def create_explainer_plugin(name: str, explain_fn: Callable) -> PluginInterface:
        """Create an explainer plugin.
        
        Args:
            name: Plugin name.
            explain_fn: Function that computes explanations.
            
        Returns:
            A PluginInterface implementation.
        """
        logger.info(f"Creating explainer plugin: {name}")
        return FunctionPluginAdapter(name, explain_fn)
        
    @staticmethod
    def create_metric_plugin(name: str, metric_fn: Callable) -> PluginInterface:
        """Create a metric plugin.
        
        Args:
            name: Plugin name.
            metric_fn: Function that computes metrics.
            
        Returns:
            A PluginInterface implementation.
        """
        logger.info(f"Creating metric plugin: {name}")
        return FunctionPluginAdapter(name, metric_fn)
        
    @staticmethod
    def create_deployment_plugin(name: str, deploy_fn: Callable) -> PluginInterface:
        """Create a deployment plugin.
        
        Args:
            name: Plugin name.
            deploy_fn: Function that handles deployment.
            
        Returns:
            A PluginInterface implementation.
        """
        logger.info(f"Creating deployment plugin: {name}")
        return FunctionPluginAdapter(name, deploy_fn)
