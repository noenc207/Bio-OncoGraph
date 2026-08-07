"""End-to-end integration tests for PyTorch backend."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tempfile
import os

# --- Mocks and Dummies ---

class MockDataset(Dataset):
    """Mock dataset for integration test."""
    def __init__(self, size: int = 50):
        self.size = size
        self.features = torch.randn(size, 20)
        self.labels = torch.randint(0, 2, (size,)).float()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

class MockGraphPreprocess:
    """Mock preprocessing step."""
    @staticmethod
    def process(features: torch.Tensor) -> torch.Tensor:
        # Simulate some graph feature extraction/normalization
        return features / (torch.norm(features, dim=1, keepdim=True) + 1e-6)

class MockGraphNetwork(nn.Module):
    """Mock graph neural network."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x).squeeze(-1)

# --- Integration Tests ---

def test_integration_end_to_end_training_loop() -> None:
    """Test full training loop: Dataset -> DataLoader -> Preprocess -> Graph -> Loss -> Backward -> Optimizer."""
    # 1. Dataset & DataLoader
    dataset = MockDataset(size=32)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 2. Model, Optimizer, Loss
    model = MockGraphNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    initial_weights = model.fc1.weight.clone()
    
    # 3. Training Loop
    model.train()
    for batch_features, batch_labels in dataloader:
        # Preprocessing (mock)
        processed_features = MockGraphPreprocess.process(batch_features)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(processed_features)
        
        # Loss calculation
        loss = criterion(predictions, batch_labels)
        
        # Backward pass & Optimization
        loss.backward()
        optimizer.step()
        
    # Verify weights were updated
    assert not torch.allclose(model.fc1.weight, initial_weights)

def test_integration_checkpoint_save_load() -> None:
    """Test training, saving checkpoint, and resuming."""
    dataset = MockDataset(size=16)
    dataloader = DataLoader(dataset, batch_size=8)
    
    model = MockGraphNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Train step 1
    features, labels = next(iter(dataloader))
    optimizer.zero_grad()
    loss = nn.BCELoss()(model(features), labels)
    loss.backward()
    optimizer.step()
    
    # Save Checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.pt")
        state = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict()
        }
        torch.save(state, ckpt_path)
        
        # Create new model & optimizer
        new_model = MockGraphNetwork()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
        
        # Load Checkpoint
        loaded_state = torch.load(ckpt_path)
        new_model.load_state_dict(loaded_state['model'])
        new_optimizer.load_state_dict(loaded_state['opt'])
        
        # Verify restored state
        assert torch.allclose(model.fc1.weight, new_model.fc1.weight)

def test_integration_evaluation_loop() -> None:
    """Test model evaluation mode and inference."""
    dataset = MockDataset(size=16)
    dataloader = DataLoader(dataset, batch_size=16)
    model = MockGraphNetwork()
    
    model.eval()
    with torch.no_grad():
        features, labels = next(iter(dataloader))
        processed = MockGraphPreprocess.process(features)
        preds = model(processed)
        
    assert preds.shape == (16,)
    assert torch.all((preds >= 0.0) & (preds <= 1.0))

def test_integration_learning_rate_scheduling() -> None:
    """Test integration of optimizer and scheduler across epochs."""
    model = MockGraphNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Simulate 1 epoch
    optimizer.step()
    scheduler.step()
    
    new_lr = optimizer.param_groups[0]['lr']
    assert new_lr == initial_lr * 0.5

def test_integration_loss_convergence() -> None:
    """Test that a simple mock model can overfit a small batch (convergence check)."""
    torch.manual_seed(42)
    features = torch.randn(4, 20)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    
    model = MockGraphNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()
    
    initial_loss = criterion(model(features), labels).item()
    
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
        
    final_loss = criterion(model(features), labels).item()
    assert final_loss < initial_loss
    assert final_loss < 0.1  # Should converge nicely on small data

def test_integration_device_placement() -> None:
    """Test pipeline works with explicit device placement (CPU)."""
    device = torch.device('cpu')
    model = MockGraphNetwork().to(device)
    
    features = torch.randn(4, 20).to(device)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0]).to(device)
    
    preds = model(features)
    loss = nn.BCELoss()(preds, labels)
    
    assert preds.device.type == 'cpu'
    assert loss.device.type == 'cpu'

def test_integration_numerical_stability() -> None:
    """Test that pipeline does not produce NaNs under extreme inputs."""
    model = MockGraphNetwork()
    # Provide extreme values
    features = torch.full((4, 20), 1e6)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    
    processed = MockGraphPreprocess.process(features)
    preds = model(processed)
    
    assert not torch.isnan(preds).any()
    assert not torch.isinf(preds).any()
