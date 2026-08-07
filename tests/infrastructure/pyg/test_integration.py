import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

class SimpleMockModel(nn.Module):
    """Mock integration model mapping GNN -> Pool -> Predict."""
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 32)
        self.classifier = nn.Linear(32, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.classifier(x)

def test_integration_framework_graph_to_pyg_data():
    """Test mapping custom dict to PyG Data."""
    raw_graph = {"nodes": torch.randn(5, 16), "edges": torch.tensor([[0, 1], [1, 2]])}
    data = Data(x=raw_graph["nodes"], edge_index=raw_graph["edges"])
    assert data.x.shape == (5, 16)
    assert data.edge_index.shape == (2, 2)

def test_integration_pyg_data_batching():
    """Test batching multiple PyG Data objects."""
    data_list = [
        Data(x=torch.randn(3, 16), edge_index=torch.tensor([[0, 1], [1, 2]])),
        Data(x=torch.randn(4, 16), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
    ]
    batch = Batch.from_data_list(data_list)
    assert batch.num_graphs == 2
    assert batch.x.shape == (7, 16)

def test_integration_message_passing_forward():
    """Test MPNN forward pass within the model."""
    model = SimpleMockModel()
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    batch = torch.tensor([0, 0, 0, 0, 0])
    out = model.conv1(x, edge_index)
    assert out.shape == (5, 32)

def test_integration_pooling_forward():
    """Test pooling forward pass within the integration loop."""
    x = torch.randn(5, 32)
    batch = torch.tensor([0, 0, 1, 1, 1])
    out = global_mean_pool(x, batch)
    assert out.shape == (2, 32)

def test_integration_prediction_head():
    """Test classification head."""
    model = SimpleMockModel()
    pooled_x = torch.randn(2, 32)
    out = model.classifier(pooled_x)
    assert out.shape == (2, 2)

def test_integration_loss_computation():
    """Test standard loss computation."""
    model = SimpleMockModel()
    out = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    target = torch.tensor([0, 1])
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(out, target)
    assert loss.item() > 0

def test_integration_backward_pass():
    """Test backward pass generation."""
    model = SimpleMockModel()
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    batch = torch.tensor([0, 0, 0, 0, 0])
    target = torch.tensor([1])
    
    out = model(x, edge_index, batch)
    loss = nn.CrossEntropyLoss()(out, target)
    loss.backward()
    
    assert model.classifier.weight.grad is not None

def test_integration_checkpoint_save(tmp_path):
    """Test checkpointing model state."""
    model = SimpleMockModel()
    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    assert ckpt_path.exists()

def test_integration_checkpoint_load(tmp_path):
    """Test loading checkpoint model state."""
    model1 = SimpleMockModel()
    model2 = SimpleMockModel()
    ckpt_path = tmp_path / "model.pt"
    torch.save(model1.state_dict(), ckpt_path)
    model2.load_state_dict(torch.load(ckpt_path))
    assert torch.equal(model1.classifier.weight, model2.classifier.weight)

def test_integration_explainability_mock_init():
    """Test mock explainability module."""
    class MockExplainer:
        def __init__(self, model):
            self.model = model
    explainer = MockExplainer(SimpleMockModel())
    assert explainer.model is not None

def test_integration_explainability_mock_forward():
    """Test mock explainability attribution."""
    class MockExplainer:
        def explain(self, x, edge_index):
            return torch.ones(edge_index.shape[1])
    explainer = MockExplainer()
    attr = explainer.explain(torch.randn(3, 16), torch.tensor([[0, 1], [1, 2]]))
    assert attr.shape == (2,)

def test_integration_full_training_loop_one_epoch():
    """Test full training loop for one epoch."""
    model = SimpleMockModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.randn(10, 16)
    edge_index = torch.randint(0, 10, (2, 15))
    batch = torch.tensor([0]*5 + [1]*5)
    target = torch.tensor([0, 1])
    
    optimizer.zero_grad()
    out = model(x, edge_index, batch)
    loss = nn.CrossEntropyLoss()(out, target)
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0

def test_integration_full_inference_loop():
    """Test full inference loop."""
    model = SimpleMockModel()
    model.eval()
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    batch = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        out = model(x, edge_index, batch)
    assert out.shape == (2, 2)

def test_integration_end_to_end_pipeline():
    """Test pipeline integration representation."""
    raw_data = {"x": torch.randn(5, 16), "edge_index": torch.tensor([[0, 1], [1, 2]])}
    data = Data(**raw_data)
    batch = Batch.from_data_list([data, data])
    
    model = SimpleMockModel()
    out = model(batch.x, batch.edge_index, batch.batch)
    
    assert out.shape == (2, 2)
