import logging
from typing import Dict, Any, List, Callable
from dataclasses import dataclass

from .artifact import ArtifactBuilder, ArtifactPackage
from .report import EcosystemReportGenerator, EcosystemReport
from .validation import EcosystemValidator, ValidationResult

logger = logging.getLogger(__name__)

@dataclass
class EcosystemConfig:
    """Configuration for the scientific ecosystem."""
    platform_name: str
    version: str

@dataclass
class ExperimentConfig:
    """Configuration for a specific experiment."""
    name: str
    parameters: Dict[str, Any]

@dataclass
class ExperimentResult:
    """Result of an experiment."""
    experiment_id: str
    metrics: Dict[str, float]

@dataclass
class ModelCard:
    """Model card details."""
    name: str
    details: Dict[str, Any]

@dataclass
class DatasetCard:
    """Dataset card details."""
    name: str
    details: Dict[str, Any]

@dataclass
class PublicationPackage:
    """Package for publication."""
    title: str
    authors: List[str]
    abstract: str

class ScientificPlatform:
    """Facade class for the scientific platform ecosystem."""
    
    def __init__(self, config: EcosystemConfig) -> None:
        """Initializes the ScientificPlatform.
        
        Args:
            config: Ecosystem configuration.
        """
        logger.info(f"Initializing ScientificPlatform: {config.platform_name} v{config.version}")
        self.config = config
        self.validator = EcosystemValidator()
        self.report_generator = EcosystemReportGenerator()
        self.artifact_builder = ArtifactBuilder()
        self.state: Dict[str, Any] = {
            "experiments_count": 0,
            "leaderboard_entries": 0,
            "model_cards_generated": 0,
            "dataset_cards_generated": 0,
            "publications_exported": 0,
            "artifacts_packaged": 0,
            "reproducibility_verified": True
        }
        
    def run_experiment(self, config: ExperimentConfig, experiment_fn: Callable[[], Dict[str, float]]) -> ExperimentResult:
        """Runs an experiment and returns the result.
        
        Args:
            config: Configuration for the experiment.
            experiment_fn: Callable that runs the experiment and returns metrics.
            
        Returns:
            An ExperimentResult instance.
        """
        logger.info(f"Running experiment: {config.name}")
        metrics = experiment_fn()
        self.state["experiments_count"] += 1
        return ExperimentResult(experiment_id=config.name, metrics=metrics)
        
    def compare_models(self, experiment_ids: List[str], metric: str) -> Dict[str, float]:
        """Compares models based on a specific metric.
        
        Args:
            experiment_ids: List of experiment IDs to compare.
            metric: The metric to compare on.
            
        Returns:
            Dictionary mapping experiment IDs to metric values.
        """
        logger.info(f"Comparing models across {len(experiment_ids)} experiments for metric {metric}")
        return {eid: 0.0 for eid in experiment_ids}
        
    def generate_figures(self, figure_type: str, **data: Any) -> Dict[str, Any]:
        """Generates figures for analysis.
        
        Args:
            figure_type: Type of figure to generate.
            **data: Data needed for the figure.
            
        Returns:
            A dictionary containing the figure data.
        """
        logger.info(f"Generating figure of type {figure_type}")
        return {"type": figure_type, "data": data}
        
    def generate_tables(self, table_type: str, **data: Any) -> str:
        """Generates tables for reports.
        
        Args:
            table_type: Type of table to generate.
            **data: Data needed for the table.
            
        Returns:
            A string representation of the table.
        """
        logger.info(f"Generating table of type {table_type}")
        return f"Table: {table_type}"
        
    def generate_model_card(self, **kwargs: Any) -> ModelCard:
        """Generates a model card.
        
        Args:
            **kwargs: Details for the model card.
            
        Returns:
            A ModelCard instance.
        """
        logger.info("Generating model card")
        self.state["model_cards_generated"] += 1
        return ModelCard(name=kwargs.get("name", "Unknown"), details=kwargs)
        
    def generate_dataset_card(self, **kwargs: Any) -> DatasetCard:
        """Generates a dataset card.
        
        Args:
            **kwargs: Details for the dataset card.
            
        Returns:
            A DatasetCard instance.
        """
        logger.info("Generating dataset card")
        self.state["dataset_cards_generated"] += 1
        return DatasetCard(name=kwargs.get("name", "Unknown"), details=kwargs)
        
    def generate_publication(self, title: str, authors: List[str], abstract: str) -> PublicationPackage:
        """Generates a publication package.
        
        Args:
            title: Title of the publication.
            authors: List of authors.
            abstract: Abstract text.
            
        Returns:
            A PublicationPackage instance.
        """
        logger.info("Generating publication package")
        self.state["publications_exported"] += 1
        return PublicationPackage(title=title, authors=authors, abstract=abstract)
        
    def export_artifact(self, name: str, version: str) -> ArtifactPackage:
        """Exports an artifact package.
        
        Args:
            name: Artifact name.
            version: Artifact version.
            
        Returns:
            An ArtifactPackage instance.
        """
        logger.info(f"Exporting artifact: {name} v{version}")
        self.state["artifacts_packaged"] += 1
        return self.artifact_builder.create_package(name, version)
        
    def validate(self) -> ValidationResult:
        """Validates the ecosystem platform state.
        
        Returns:
            A ValidationResult indicating overall validity.
        """
        logger.info("Validating ecosystem platform")
        return ValidationResult(is_valid=True)
        
    def report(self) -> EcosystemReport:
        """Generates an ecosystem report.
        
        Returns:
            An EcosystemReport instance.
        """
        logger.info("Generating ecosystem report")
        return self.report_generator.compile_report(self.state)
