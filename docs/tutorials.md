# Tutorials

Welcome to the BioOncoGraph tutorials. These guides provide step-by-step instructions for common tasks.

## Tutorial 1: Creating a Custom Dataset

This tutorial explains how to convert a custom CSV file of drug-target interactions into an `OncoGraph` object.

### Step 1: Define the Dataset Class

```python
import pandas as pd
import torch
from biooncograph.data import Dataset, OncoGraph

class MyCustomDTI(Dataset):
    def __init__(self, root, csv_path):
        self.csv_path = csv_path
        super().__init__(root)

    @property
    def raw_file_names(self):
        return ['custom_dti.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # 1. Read the data
        df = pd.read_csv(self.csv_path)

        # 2. Extract unique entities
        drugs = df['drug_id'].unique()
        targets = df['target_id'].unique()

        # Create mapping dicts
        drug_map = {id: i for i, id in enumerate(drugs)}
        target_map = {id: i for i, id in enumerate(targets)}

        # 3. Create Edge Index
        src = [drug_map[id] for id in df['drug_id']]
        dst = [target_map[id] for id in df['target_id']]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # 4. Construct the Graph
        graph = OncoGraph()
        graph['drug'].num_nodes = len(drugs)
        graph['target'].num_nodes = len(targets)
        graph['drug', 'interacts_with', 'target'].edge_index = edge_index

        # 5. Save the graph
        torch.save(graph, self.processed_paths[0])
```

## Tutorial 2: Using the Multimodal Feature Fusion

BioOncoGraph allows combining different types of embeddings.

```python
import biooncograph as bog

# Assume graph has 'drug' and 'gene' nodes
graph = bog.datasets.SampleOncologyGraph()[0]

# 1. Initialize Encoders
prot_encoder = bog.features.ProteinEncoder(model_name='protbert')
mol_encoder = bog.features.MoleculeEncoder(method='morgan_fingerprints')

# 2. Generate Features (Assuming raw sequences/SMILES are stored in the graph)
gene_features = prot_encoder.encode(graph['gene'].sequence)
drug_features = mol_encoder.encode(graph['drug'].smiles)

# 3. Assign features back to the graph
graph['gene'].x = gene_features
graph['drug'].x = drug_features

# 4. Proceed with training...
```

## More Tutorials Coming Soon
*   Explainability and Subgraph Discovery
*   Integrating Large Language Models (LLMs) with the Graph
*   Deploying to AWS SageMaker
