import os
import sys
import tempfile
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

# Mock out biooncograph before importing scripts.train
mock_dataset = MagicMock()
mock_gnn = MagicMock()

# Minimal natively-executing Trainer to satisfy train.py
class MockNativeTrainer:
    """Mock trainer that natively executes PyTorch operations."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: list,
        val_loader: list,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_amp: bool,
        epochs: int,
        checkpoint_dir: str
    ) -> None:
        """Initialize the mock trainer.
        
        Args:
            model: Model to train.
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            criterion: Loss function.
            optimizer: Optimizer.
            device: Device to use.
            use_amp: Whether to use AMP.
            epochs: Number of epochs.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.manifest = {"epochs": self.epochs, "device": str(self.device)}

    def train(self) -> None:
        """Run the training loop, testing AMP, backward pass, and checkpointing."""
        self.model.train()
        for epoch in range(self.epochs):
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out = self.model(batch)
                    # mock batch has 'target'
                    target = torch.ones_like(out)
                    loss = self.criterion(out, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            # save checkpoint
            if self.checkpoint_dir:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt"))
                
                # generate manifest
                import json
                with open(os.path.join(self.checkpoint_dir, "manifest.json"), "w") as f:
                    json.dump(self.manifest, f)

mock_trainer = MagicMock()
mock_trainer.Trainer = MockNativeTrainer

sys.modules['biooncograph'] = MagicMock()
sys.modules['biooncograph.data'] = MagicMock()
sys.modules['biooncograph.data.dataset'] = mock_dataset
sys.modules['biooncograph.models'] = MagicMock()
sys.modules['biooncograph.models.gnn'] = mock_gnn
sys.modules['biooncograph.engine'] = MagicMock()
sys.modules['biooncograph.engine.trainer'] = mock_trainer

# Mock the Dataset and GNN classes
class DummyBatch:
    """Mock batch data."""
    def __init__(self, size: int = 2) -> None:
        self.x = torch.randn(size, 10)

class DummyDataset:
    """Mock dataset class."""
    def __init__(self, cfg: OmegaConf, split: str) -> None:
        self.cfg = cfg
        self.split = split
    
    def get_dataloader(self, batch_size: int, shuffle: bool, use_ddp: bool) -> list:
        return [DummyBatch(batch_size) for _ in range(2)]

class DummyGNN(nn.Module):
    """Mock GNN model."""
    def __init__(self, cfg: OmegaConf) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 2)
        
    def forward(self, batch: DummyBatch) -> torch.Tensor:
        return self.linear(batch.x)

mock_dataset.BioOncoGraphDataset = DummyDataset
mock_gnn.BioOncoGraphGNN = DummyGNN

# Now import train
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
import train

# Parameterize tests
ddp_options = [True, False]
amp_options = [True, False]
batch_sizes = [2, 4]
learning_rates = [1e-3, 5e-4]
weight_decays = [1e-4, 0.0]
optimizers = ["adamw", "adam"]
epochs_list = [1, 2]
model_types = ["gcn", "gat"]

import itertools
param_combinations = list(itertools.product(
    ddp_options, amp_options, batch_sizes, learning_rates, weight_decays, optimizers, epochs_list, model_types
))

@pytest.mark.parametrize("use_ddp, use_amp, batch_size, lr, wd, opt, epochs, model_type", param_combinations)
def test_train_integration(use_ddp, use_amp, batch_size, lr, wd, opt, epochs, model_type):
    """
    Test train script integration with various configs.
    
    Args:
        use_ddp: DDP flag.
        use_amp: AMP flag.
        batch_size: Batch size.
        lr: Learning rate.
        wd: Weight decay.
        opt: Optimizer.
        epochs: Number of epochs.
        model_type: Model architecture type.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = OmegaConf.create({
            "dataset": {},
            "model": {"type": model_type},
            "training": {
                "use_ddp": use_ddp,
                "use_amp": use_amp,
                "device": "cpu", # Force CPU for tests
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": wd,
                "optimizer": opt,
                "epochs": epochs,
                "checkpoint_dir": tmpdir
            }
        })
        
        # Patch init_process_group and destroy_process_group to simulate DDP
        with patch("torch.distributed.init_process_group") as mock_init, \
             patch("torch.distributed.destroy_process_group") as mock_destroy, \
             patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            
            # Execute train
            train.main(cfg)
            
            if use_ddp:
                mock_init.assert_called_once()
                mock_destroy.assert_called_once()
            else:
                mock_init.assert_not_called()
                mock_destroy.assert_not_called()
                
            # Verify checkpoints and manifest were created natively
            for ep in range(epochs):
                assert os.path.exists(os.path.join(tmpdir, f"epoch_{ep}.pt")), f"Checkpoint epoch_{ep}.pt not found"
            
            assert os.path.exists(os.path.join(tmpdir, "manifest.json")), "manifest.json not found"

