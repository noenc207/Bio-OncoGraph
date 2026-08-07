# Examples

This page contains quick, copy-pasteable code snippets for common operations in BioOncoGraph v2.0.

### 1. Zero-Shot Inference with Pre-trained Models

Use a foundation model to predict interactions without any training.

```python
import biooncograph as bog

model = bog.models.BioModel.from_pretrained('bog-v2-foundation-large')
model.eval()

# Check if a drug interacts with a specific mutated gene
score = bog.infer.check_interaction(
    model=model,
    entity1=('drug', 'Aspirin'),
    entity2=('gene', 'TP53_mutant')
)
print(f"Interaction Score: {score}")
```

### 2. Exporting Graph to NetworkX for Visualization

Sometimes you need to visualize a small subgraph using standard tools.

```python
import biooncograph as bog
import networkx as nx
import matplotlib.pyplot as plt

graph = bog.datasets.SampleOncologyGraph()[0]

# Extract a homogeneous subgraph for simpler visualization
subgraph = graph.to_homogeneous()

# Convert PyG homogeneous data to NetworkX
nx_graph = bog.utils.to_networkx(subgraph, to_undirected=True)

plt.figure(figsize=(8,8))
nx.draw(nx_graph, node_size=10, node_color='red')
plt.show()
```

### 3. Hyperparameter Optimization (Optuna Integration)

BioOncoGraph integrates easily with tools like Optuna.

```python
import optuna
import biooncograph as bog

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])

    model = bog.models.build_default_dti_model(hidden_dim=hidden_dim)
    trainer = bog.engine.Trainer(model, lr=lr, max_epochs=10)

    # Train and return validation metric
    trainer.fit(train_loader, val_loader)
    return trainer.callback_metrics['val_auroc'].item()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(study.best_params)
```

### 4. Querying the internal database

```python
from biooncograph.data import Database

db = Database.connect()
genes = db.query_nodes(type="gene", pathway="MAPK")
print(f"Found {len(genes)} genes in the MAPK pathway.")
```
