# Frequently Asked Questions (FAQ)

## General

**Q: What exactly is a "multimodal knowledge graph" in this context?**
A: It is a graph data structure where nodes represent diverse biological entities (e.g., genes, proteins, drugs, patients, pathways) and edges represent known or inferred relationships (e.g., drug-targets-protein, patient-has-mutation). "Multimodal" refers to the integration of different data types (genomics, clinical notes, chemical structures) as features on these nodes.

**Q: Is BioOncoGraph approved for clinical use?**
A: **No.** BioOncoGraph is currently a research tool designed for exploratory analysis, target discovery, and computational biology. It is not approved by the FDA or any other regulatory body for diagnostic or therapeutic decision-making.

## Technical & Installation

**Q: What are the hardware requirements for training the full model?**
A: Training the full MRT-GNN on the complete dataset requires significant compute. We recommend at least a node with 4x NVIDIA A100 or H100 GPUs (80GB VRAM each) and 512GB of system RAM. 

**Q: Can I run inference on a single GPU or CPU?**
A: Yes! Inference requires far less memory. A standard GPU (e.g., RTX 3090 or 4090 with 24GB VRAM) is sufficient for querying the graph and generating predictions. CPU inference is supported but will be considerably slower.

**Q: Do you support DGL (Deep Graph Library)?**
A: BioOncoGraph v1.x utilized DGL. However, for v2.0, we have completely migrated to **PyTorch Geometric (PyG)** to leverage its extensive distributed training capabilities and tighter integration with the PyTorch ecosystem.

## Data

**Q: How do I access the raw TCGA or DepMap data?**
A: You must ensure you comply with the respective data use agreements of TCGA and DepMap. We provide a script (`scripts/download_datasets.py`) that interacts with public APIs to fetch open-access data, but restricted tier data requires your own authentication tokens.

**Q: Can I add my own proprietary hospital data to the graph?**
A: Yes. The modular design of the `DataIngestionPipeline` allows you to define custom Node and Edge schemas to seamlessly merge local, proprietary subgraphs with the public, pre-trained knowledge base.
