import dataclasses
import datetime
import logging
from typing import Dict, Any

from .utils import generate_timestamp
from .validation import OpenScienceValidator

logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class OpenScienceReport:
    """A comprehensive report of the open science platform state."""
    documentation_pages: int
    examples_count: int
    plugins_count: int
    benchmarks_count: int
    templates_count: int
    reproducibility_valid: bool
    timestamp: str

class ReportGenerator:
    """Generator for open science reports."""
    
    def __init__(self) -> None:
        self._validator = OpenScienceValidator()
        
    def compile_report(self, platform_state: Dict[str, Any]) -> OpenScienceReport:
        """Compile a report from the current platform state.
        
        Args:
            platform_state: Dictionary containing platform components and metrics.
            
        Returns:
            A populated OpenScienceReport.
        """
        logger.info("Compiling open science report.")
        
        # Extract metrics safely from state
        docs = platform_state.get("documentation", [])
        examples = platform_state.get("examples", [])
        plugins = platform_state.get("plugins", [])
        benchmarks = platform_state.get("benchmarks", [])
        templates = platform_state.get("templates", [])
        repro_pkg = platform_state.get("reproducibility_package")
        
        # Validate reproducibility package
        repro_valid = self._validator.validate_reproducibility(repro_pkg).is_valid
        
        report = OpenScienceReport(
            documentation_pages=len(docs) if isinstance(docs, list) else 0,
            examples_count=len(examples) if isinstance(examples, list) else 0,
            plugins_count=len(plugins) if isinstance(plugins, list) else 0,
            benchmarks_count=len(benchmarks) if isinstance(benchmarks, list) else 0,
            templates_count=len(templates) if isinstance(templates, list) else 0,
            reproducibility_valid=repro_valid,
            timestamp=generate_timestamp()
        )
        logger.info("Open science report compiled successfully.")
        return report
