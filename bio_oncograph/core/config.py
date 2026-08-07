"""
Hydra Structured Config schemas for Bio-OncoGraph.

Every configuration group is defined as a frozen ``@dataclass`` with
explicit types.  OmegaConf validates types and mandatory fields at
parse time, enforcing the Fail-Fast engineering constraint.

Design references:
  - API Bible § 8 (Config Schema)
  - API Bible § 40 (Backward Compatibility)
  - ADR-010 (Config Management)
"""

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class ModelConfig:
    """
    Configuration for the Bio-OncoGraph model architecture.

    Attributes:
        in_channels: Dimensionality of input node features (e.g. 1024
            for UNI, 768 for CTransPath).  Mandatory — no default.
        hidden_dim: Hidden dimension throughout the GNN layers.
        num_phenotypes: Number of pseudo-phenotype clusters in
            PhenoGraph.
        num_relations: Number of relation types in the multi-relational
            graph.
        pool_ratios: Fraction of nodes retained at each pooling stage.
        num_mp_layers: Number of BioMP message-passing rounds per
            resolution level.
        heads: Number of attention heads in multi-head attention layers.
        dropout: Dropout probability applied after each GNN layer.
        time_bins: Number of discrete survival time bins.
        tau_phenograph: Temperature for the PhenoGraph softmax.
        tau_topopool: Temperature for the TopoPool filtration sigmoid.
    """

    in_channels: int = MISSING
    hidden_dim: int = 256
    num_phenotypes: int = 8
    num_relations: int = 4
    pool_ratios: List[float] = field(default_factory=lambda: [0.5, 0.25])
    num_mp_layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    time_bins: int = 4
    tau_phenograph: float = 1.0
    tau_topopool: float = 1.0


@dataclass
class DatasetConfig:
    """
    Configuration for dataset paths and preprocessing.

    Attributes:
        name: Identifier for the dataset (e.g. ``tcga_brca``).
        root_dir: Root directory containing processed graph features.
            Mandatory — no default.
        split_dir: Directory containing train/val/test split CSV files.
        feature_extractor: Name of the encoder used for patch features.
        k_neighbours: Number of nearest neighbours for graph
            construction.
        max_nodes: Maximum number of nodes per graph.  Graphs exceeding
            this limit are randomly sub-sampled.  Set to 0 to disable.
    """

    name: str = MISSING
    root_dir: str = MISSING
    split_dir: str = MISSING
    feature_extractor: str = "uni_v1"
    k_neighbours: int = 8
    max_nodes: int = 0


@dataclass
class LossConfig:
    """
    Configuration for the CalTopoSurv composite loss.

    Attributes:
        alpha: Weight for the ranking (concordance) loss component.
        beta: Weight for the topological (SWD) loss component.
        num_projections: Number of random projections for
            Sliced-Wasserstein Distance computation.
        epsilon: Small constant added to denominators and square roots
            for numerical stability.
    """

    alpha: float = 1.0
    beta: float = 0.5
    num_projections: int = 50
    epsilon: float = 1e-8


@dataclass
class OptimizerConfig:
    """
    Configuration for the optimiser.

    Attributes:
        name: Optimiser class name (``adam``, ``adamw``).
        lr: Learning rate.
        weight_decay: L2 regularisation coefficient.
    """

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-5


@dataclass
class SchedulerConfig:
    """
    Configuration for the learning-rate scheduler.

    Attributes:
        name: Scheduler type (``cosine``, ``step``, ``none``).
        warmup_epochs: Number of linear warm-up epochs.
        min_lr: Minimum learning rate for cosine annealing.
    """

    name: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    """
    Configuration for the training engine.

    Attributes:
        epochs: Total number of training epochs.
        batch_size: Number of graphs per mini-batch.
        num_workers: DataLoader worker count.
        pin_memory: Whether to pin host memory for faster GPU transfer.
        gradient_accumulation_steps: Number of forward passes before a
            single ``optimizer.step()``.
        gradient_clip_max_norm: Maximum gradient norm for clipping.
            Set to 0.0 to disable.
        early_stopping_patience: Number of epochs without improvement
            before stopping.  Set to 0 to disable.
        optimizer: Optimiser sub-config.
        scheduler: Scheduler sub-config.
    """

    epochs: int = 100
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip_max_norm: float = 1.0
    early_stopping_patience: int = 10
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class LoggerConfig:
    """
    Configuration for experiment logging.

    Attributes:
        project: WandB project name.
        name: Run name.
        use_wandb: Whether to activate WandB logging.
    """

    project: str = "bio-oncograph"
    name: str = "experiment_run"
    use_wandb: bool = False


@dataclass
class ExperimentConfig:
    """
    Top-level experiment configuration aggregating all sub-configs.

    This dataclass mirrors the structure of ``configs/config.yaml``
    and is the root node of the Hydra configuration tree.

    Attributes:
        seed: Global random seed for reproducibility.
        run_mode: Execution mode (TRAIN, EVAL, INFERENCE, EXPLAIN).
        device: Target hardware device (CUDA, CPU, MPS).
        amp: Whether to enable Automatic Mixed Precision.
        output_dir: Directory for checkpoints, logs, and artifacts.
        model: Model architecture sub-config.
        dataset: Dataset sub-config.
        loss: Loss function sub-config.
        training: Training engine sub-config.
        logger: Logger sub-config.
    """

    seed: int = 42
    run_mode: str = "TRAIN"
    device: str = "CUDA"
    amp: bool = True
    output_dir: str = "outputs"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)

@dataclass
class TrainingEngineConfig:
    amp: bool = True
    ddp: bool = False
    gradient_accumulation_steps: int = 1
    gradient_clip_val: float = 1.0
    deterministic: bool = False
