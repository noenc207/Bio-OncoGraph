# API Reference

This page provides an overview of the core API for BioOncoGraph v2.0. For complete class and method details, please refer to the Python docstrings or the generated Sphinx documentation.

## `biooncograph.data`

Core structures for managing biological data.

*   **`OncoGraph`**: The central data structure representing a heterogeneous knowledge graph. Inherits from PyG's `HeteroData`.
*   **`Dataset`**: Base class for all datasets.
*   **`DataLoader` / `HeteroLinkDataloader`**: Highly optimized loaders for mini-batching large graphs.

## `biooncograph.models`

Neural network architectures.

*   **`BioModel`**: The main wrapper class combining an encoder and decoder. Provides `from_pretrained()` and `save_pretrained()` methods.
*   **`HeteroGNN`**: Configurable Heterogeneous Graph Neural Network encoder (supports HGT, RGCN, GAT).
*   **`LinkPredictor`**: Decoder for edge prediction tasks.
*   **`NodeClassifier`**: Decoder for node classification tasks.

## `biooncograph.engine`

Execution and lifecycle management.

*   **`Trainer`**: Manages the training loop, distributed execution, and checkpointing.
    *   `Trainer.fit(train_loader, val_loader)`
    *   `Trainer.test(test_loader)`
*   **`Evaluator`**: Computes task-specific metrics.
    *   `Evaluator.compute(predictions, labels)`

## `biooncograph.features`

Modules for generating feature embeddings.

*   **`ProteinEncoder`**: Wraps pre-trained models like ProtBERT to generate protein sequence embeddings.
*   **`MoleculeEncoder`**: Generates embeddings from SMILES strings or 3D conformations.
*   **`ClinicalTextEncoder`**: Utilizes LLMs to encode clinical notes into node features.

## `biooncograph.apps`

High-level wrappers for the 14 research tracks.

*   `DTIApp` (Track 1: Drug-Target Interaction)
*   `DRPApp` (Track 2: Drug Response Prediction)
*   `BiomarkerApp` (Track 4: Biomarker Discovery)
*   *(... and 11 others)*

## `biooncograph.transforms`

Data augmentation and splitting strategies.

*   **`RandomLinkSplit`**: Standard random splitting for edges.
*   **`ScaffoldSplit`**: Splits chemical compounds based on structural scaffolds.
*   **`TemporalSplit`**: Splits edges based on timestamp attributes.

## Global Configuration

*   **`bog.config.set_device(device)`**: Set global execution device.
*   **`bog.config.set_seed(seed)`**: Set random seed for reproducibility.
