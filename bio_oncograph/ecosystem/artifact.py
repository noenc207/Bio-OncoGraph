import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ArtifactPackage:
    """Represents a packaged artifact for distribution or archival.
    
    Attributes:
        name: Name of the package.
        version: Version of the package.
        source_files: Dictionary mapping file paths to file contents.
        configs: Dictionary of configuration settings.
        checkpoints: Dictionary mapping checkpoint names to data.
        scripts: Dictionary of script contents.
        documentation: Dictionary of documentation strings.
        metadata: General metadata about the artifact.
    """
    name: str
    version: str
    source_files: Dict[str, str] = field(default_factory=dict)
    configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    checkpoints: Dict[str, bytes] = field(default_factory=dict)
    scripts: Dict[str, str] = field(default_factory=dict)
    documentation: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ArtifactBuilder:
    """Builds and manages artifact packages."""
    
    def create_package(self, name: str, version: str) -> ArtifactPackage:
        """Creates a new, empty artifact package.
        
        Args:
            name: Package name.
            version: Package version.
            
        Returns:
            A new ArtifactPackage instance.
        """
        logger.info(f"Creating artifact package: {name} v{version}")
        return ArtifactPackage(name=name, version=version)
        
    def add_source(self, package: ArtifactPackage, file_path: str, content: str) -> ArtifactPackage:
        """Adds a source file to an artifact package.
        
        Args:
            package: The original ArtifactPackage.
            file_path: Path to the source file.
            content: Content of the source file.
            
        Returns:
            A new ArtifactPackage with the source added.
        """
        logger.info(f"Adding source to {package.name}: {file_path}")
        sources = dict(package.source_files)
        sources[file_path] = content
        return ArtifactPackage(
            name=package.name, version=package.version, source_files=sources,
            configs=package.configs, checkpoints=package.checkpoints,
            scripts=package.scripts, documentation=package.documentation,
            metadata=package.metadata
        )
        
    def add_config(self, package: ArtifactPackage, config_name: str, config_dict: Dict[str, Any]) -> ArtifactPackage:
        """Adds a configuration to an artifact package.
        
        Args:
            package: The original ArtifactPackage.
            config_name: Name of the configuration.
            config_dict: Configuration dictionary.
            
        Returns:
            A new ArtifactPackage with the config added.
        """
        logger.info(f"Adding config to {package.name}: {config_name}")
        configs = dict(package.configs)
        configs[config_name] = config_dict
        return ArtifactPackage(
            name=package.name, version=package.version, source_files=package.source_files,
            configs=configs, checkpoints=package.checkpoints,
            scripts=package.scripts, documentation=package.documentation,
            metadata=package.metadata
        )
        
    def add_checkpoint(self, package: ArtifactPackage, checkpoint_name: str, checkpoint_data: bytes) -> ArtifactPackage:
        """Adds a binary checkpoint to an artifact package.
        
        Args:
            package: The original ArtifactPackage.
            checkpoint_name: Name of the checkpoint.
            checkpoint_data: Binary data of the checkpoint.
            
        Returns:
            A new ArtifactPackage with the checkpoint added.
        """
        logger.info(f"Adding checkpoint to {package.name}: {checkpoint_name}")
        checkpoints = dict(package.checkpoints)
        checkpoints[checkpoint_name] = checkpoint_data
        return ArtifactPackage(
            name=package.name, version=package.version, source_files=package.source_files,
            configs=package.configs, checkpoints=checkpoints,
            scripts=package.scripts, documentation=package.documentation,
            metadata=package.metadata
        )
        
    def export_manifest(self, package: ArtifactPackage) -> str:
        """Exports a JSON manifest of the artifact package.
        
        Args:
            package: The ArtifactPackage to export.
            
        Returns:
            A JSON string representing the package manifest.
        """
        logger.info(f"Exporting manifest for {package.name}")
        manifest = {
            "name": package.name,
            "version": package.version,
            "files": list(package.source_files.keys()),
            "configs": list(package.configs.keys()),
            "checkpoints": list(package.checkpoints.keys())
        }
        return json.dumps(manifest, indent=2)
        
    def validate_package(self, package: ArtifactPackage) -> List[str]:
        """Validates an artifact package.
        
        Args:
            package: The ArtifactPackage to validate.
            
        Returns:
            A list of validation error messages, empty if valid.
        """
        logger.info(f"Validating package {package.name}")
        errors = []
        if not package.name:
            errors.append("Package name is empty.")
        if not package.version:
            errors.append("Package version is empty.")
        return errors
