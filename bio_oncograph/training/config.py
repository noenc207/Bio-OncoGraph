"""Training configuration."""
import dataclasses
from typing import Optional


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for training.
    
    Attributes:
        epochs: Number of training epochs.
        batch_size: Batch size for training and validation.
        optimizer: Name or type of the optimizer to use.
        scheduler: Name or type of the learning rate scheduler.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay factor.
        seed: Random seed for reproducibility.
        mixed_precision: Whether to use mixed precision training.
        gradient_accumulation: Number of steps to accumulate gradients.
        gradient_clip: Maximum norm for gradient clipping.
        num_workers: Number of workers for data loading.
        pin_memory: Whether to pin memory in data loaders.
        persistent_workers: Whether to use persistent workers in data loaders.
        checkpoint_dir: Directory to save checkpoints.
        resume: Path to a checkpoint to resume training from.
        save_every: Save a checkpoint every N epochs.
        validate_every: Validate the model every N epochs.
    """
    epochs: int = 100
    batch_size: int = 32
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    mixed_precision: bool = False
    gradient_accumulation: int = 1
    gradient_clip: float = 1.0
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    checkpoint_dir: str = "checkpoints"
    resume: Optional[str] = None
    save_every: int = 1
    validate_every: int = 1
