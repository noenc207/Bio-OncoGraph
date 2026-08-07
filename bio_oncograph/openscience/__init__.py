"""Open Science Framework for Bio-OncoGraph.

This package provides generic open science tooling for plugins,
validation, reporting, extensions, and artifact generation.
"""

from .plugins import PluginInterface, PluginRegistry
from .sdk import PluginSDK, FunctionPluginAdapter
from .extensions import ExtensionPoint, ExtensionRegistry
from .validation import OpenScienceValidator, ValidationResult
from .report import OpenScienceReport, ReportGenerator
from .artifacts import ArtifactExporter
from .utils import (
    sanitize_module_name,
    generate_timestamp,
    compute_content_hash,
    validate_semver
)
from .platform import (
    OpenSciencePlatform,
    OpenScienceConfig,
    DocumentationIndex,
    ApiReference,
    Example,
    ReproducibilityPackage
)

__all__ = [
    "PluginInterface",
    "PluginRegistry",
    "PluginSDK",
    "FunctionPluginAdapter",
    "ExtensionPoint",
    "ExtensionRegistry",
    "OpenScienceValidator",
    "ValidationResult",
    "OpenScienceReport",
    "ReportGenerator",
    "ArtifactExporter",
    "sanitize_module_name",
    "generate_timestamp",
    "compute_content_hash",
    "validate_semver",
    "OpenSciencePlatform",
    "OpenScienceConfig",
    "DocumentationIndex",
    "ApiReference",
    "Example",
    "ReproducibilityPackage",
]
