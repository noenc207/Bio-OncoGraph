# First Training Plan

This document outlines the 10-step workflow for the initial training pipeline.

## 10-Step Workflow

1. **Deploy**: Set up the server environment (Ubuntu 24.04, CUDA, cuDNN, OpenSlide).
2. **TCGA**: Download and preprocess the target TCGA dataset.
3. **CONCH**: Load the CONCH foundation model for feature extraction.
4. **extract patch**: Tile the Whole Slide Images (WSIs) into patches, filtering out background.
5. **extract embedding**: Pass the extracted patches through the CONCH model to generate embeddings.
6. **build graph**: Construct a spatial graph where nodes are patch embeddings and edges are defined by spatial proximity.
7. **train**: Train a Graph Neural Network (GNN) on the constructed graphs for the downstream task.
8. **benchmark**: Evaluate the model against established baselines using standard metrics.
9. **ablation**: Perform ablation studies to analyze the contribution of spatial edges, embedding types, and graph architectures.
10. **paper**: Draft the manuscript detailing the methodology, results, and insights.

## Training Configuration

### Loss Function
- **Task**: Classification
- **Loss**: Cross-Entropy Loss

### Optimizer
- **Algorithm**: AdamW
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-5

### Batch Size
- **Size**: 16 to 32 graphs per batch, depending on GPU memory constraints.
