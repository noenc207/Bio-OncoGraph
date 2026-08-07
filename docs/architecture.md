# Architecture

BioOncoGraph v2.0 is built upon a **Clean Architecture** paradigm, ensuring separation of concerns, high modularity, and robust scalability. This design allows researchers to easily swap components, implement custom logic, and scale from local experiments to massive distributed training clusters.

## Core Principles

1.  **Separation of Concerns**: Data loading, graph processing, model architecture, and training logic are strictly decoupled.
2.  **Modularity**: Components interact through well-defined interfaces, making it simple to extend the framework.
3.  **Performance First**: Core operations are optimized using PyTorch Geometric and custom C++/CUDA kernels.

## System Overview

The framework is divided into five primary layers:

### 1. Data Layer (`biooncograph.data`)

Responsible for ingesting, transforming, and storing raw biological data.
*   **Parsers**: Modules to read specialized formats (e.g., VCF, FASTQ, SDF, SMILES).
*   **GraphBuilder**: Constructs the heterogeneous `OncoGraph` object.
*   **Storage**: Efficient in-memory or disk-backed (LMDB) storage for large graphs.

### 2. Feature Engineering Layer (`biooncograph.features`)

Transforms raw node/edge attributes into dense embeddings.
*   **Encoders**: Specialized encoders for different modalities (e.g., ProtBERT for proteins, Mol2Vec for drugs, LLMs for clinical text).
*   **Fusion**: Mechanisms to combine multimodal features into a unified vector representation.

### 3. Modeling Layer (`biooncograph.models`)

Contains the core Graph Neural Network architectures.
*   **GNN Layers**: Custom layers optimized for heterogeneous graphs (e.g., HGT, RGCN, GATv2).
*   **Decoders**: Task-specific output heads (link prediction, node classification, graph regression).
*   **Pre-trained Hub**: Interface for loading and saving models.

### 4. Execution Layer (`biooncograph.engine`)

Manages the training, validation, and inference lifecycles.
*   **Trainer**: Handles the training loop, gradient accumulation, and mixed precision.
*   **Samplers**: Efficient neighborhood sampling for training on massive graphs (e.g., HCSampler, RandomWalkSampler).
*   **Evaluator**: Computes specialized metrics (AUROC, AUPRC, NDCG) across different research tracks.

### 5. Application Layer (`biooncograph.apps`)

High-level APIs mapping directly to the 14 research tracks, abstracting away the underlying complexity for domain experts.

## Data Flow

1.  Raw data -> `Data Layer` -> `OncoGraph` object.
2.  `OncoGraph` -> `Feature Engineering` -> Enriched `OncoGraph` (with embeddings).
3.  Enriched `OncoGraph` -> `Execution Layer` (Samplers) -> Mini-batches.
4.  Mini-batches -> `Modeling Layer` -> Predictions.
5.  Predictions -> Loss calculation/Evaluation.
