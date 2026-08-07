"""Regression testing for Bio-OncoGraph modules."""

import importlib
import inspect
from typing import Dict, List, Optional


class RegressionTester:
    """Provides functionality to test regression and API compatibility."""

    def __init__(self) -> None:
        """Initializes the RegressionTester."""
        pass

    def check_api_compatibility(self, module_name: str, required_classes: Optional[Dict[str, List[str]]] = None) -> bool:
        """Verifies that required classes and methods exist in the given module.

        Args:
            module_name: The string name of the module to import and check.
            required_classes: A dictionary mapping class names to lists of required method names.
                If None, checks basic importability only.

        Returns:
            True if the module and all required classes/methods are found, False otherwise.
        """
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return False

        if required_classes is None:
            return True

        for class_name, methods in required_classes.items():
            if not hasattr(module, class_name):
                return False
            
            cls = getattr(module, class_name)
            if not inspect.isclass(cls):
                return False

            for method_name in methods:
                if not hasattr(cls, method_name):
                    return False
                
                method = getattr(cls, method_name)
                if not callable(method):
                    return False

        return True
