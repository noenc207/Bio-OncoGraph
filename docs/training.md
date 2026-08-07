# Training Guide

This guide details how to train models using BioOncoGraph v2.0. The framework supports both full-batch training for smaller graphs and scalable mini-batch training for massive datasets.

## 1. Preparing the Dataset

First, prepare your `OncoGraph` object. Ensure that your graph has the necessary train, validation, and test splits defined.

```python
import biooncograph as bog
from biooncograph.transforms import RandomLinkSplit

dataset = bog.datasets.TCGAMultiOmics(root='./data/tcga')
graph = dataset[0]

# Split the graph for link prediction
transform = RandomLinkSplit(
    num_val=0.1, num_test=0.1,
    is_undirected=True,
    edge_types=[('drug', 'interacts_with', 'gene')]
)
train_data, val_data, test_data = transform(graph)
```

## 2. Defining the Model

Choose a model architecture. BioOncoGraph provides flexible builders for heterogeneous GNNs.

```python
from biooncograph.models import HeteroGNN, LinkPredictor

# Define the base GNN encoder
encoder = HeteroGNN(
    metadata=train_data.metadata(),
    hidden_channels=256,
    num_layers=3,
    model_type='HGT' # Heterogeneous Graph Transformer
)

# Define the task-specific decoder
decoder = LinkPredictor(
    hidden_channels=256,
    edge_types=train_data.edge_types
)

# Combine into a single model
model = bog.models.BioModel(encoder, decoder)
model.to(bog.config.device)
```

## 3. Configuring the Trainer

The `Trainer` class handles the execution loop. It integrates seamlessly with PyTorch Lightning under the hood for distributed training and logging.

```python
from biooncograph.engine import Trainer, HeteroLinkDataloader
import torch

# Setup data loaders for mini-batch training
train_loader = HeteroLinkDataloader(train_data, batch_size=1024, shuffle=True)
val_loader = HeteroLinkDataloader(val_data, batch_size=1024)

# Define optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Initialize Trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    precision='16-mixed', # Mixed precision training
    logger='tensorboard',
    checkpoint_dir='./checkpoints'
)
```

## 4. Execution

Start the training process. The trainer will automatically handle validation and model checkpointing.

```python
trainer.fit(train_dataloader=train_loader, val_dataloader=val_loader)
```

## 5. Fine-Tuning Pre-trained Models

To fine-tune a pre-trained model on your specific dataset:

```python
model = bog.models.BioModel.from_pretrained('bog-v2-foundation')

# Freeze the encoder for linear probing
for param in model.encoder.parameters():
    param.requires_grad = False

# Proceed with Trainer as above
```

## Best Practices

*   **Sampling**: Always use neighbor sampling (`HeteroLinkDataloader` or `NeighborLoader`) when the graph exceeds GPU memory.
*   **Gradient Accumulation**: If batch sizes are small due to memory constraints, increase `accumulate_grad_batches` in the Trainer.
*   **Monitoring**: Utilize the integrated TensorBoard or Weights & Biases loggers to track metrics.
