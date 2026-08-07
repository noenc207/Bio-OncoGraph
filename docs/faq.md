# Frequently Asked Questions (FAQ)

### General

**Q: What is the difference between BioOncoGraph v1 and v2.0?**
**A:** v2.0 introduces a complete architectural rewrite focusing on the "Clean Architecture" paradigm. It includes the 14 standardized research tracks, multimodal feature fusion (integrating LLM embeddings), and a custom high-performance execution engine that reduces memory footprint by up to 60%.

**Q: Is BioOncoGraph only for oncology?**
**A:** While the pre-trained models, datasets, and the 14 research tracks are specifically curated and optimized for oncology, the underlying heterogeneous graph framework (`biooncograph.engine` and `biooncograph.models`) is domain-agnostic and can be applied to any biomedical knowledge graph.

### Installation & Setup

**Q: I get a CUDA OOM (Out of Memory) error during training. What should I do?**
**A:** Try the following steps:
1. Decrease the `batch_size` in your `HeteroLinkDataloader`.
2. Enable gradient accumulation in the `Trainer` (`accumulate_grad_batches=4`).
3. Enable mixed precision training (`precision='16-mixed'`).
4. If training on full graphs, switch to mini-batch sampling.

**Q: Can I run BioOncoGraph on a Mac with Apple Silicon (M1/M2/M3)?**
**A:** Yes, PyTorch now supports MPS (Metal Performance Shaders). BioOncoGraph will automatically detect MPS and use it if available. Note that some custom C++ operations might fallback to CPU.

### Data & Modeling

**Q: How do I add my custom dataset?**
**A:** You need to create a custom class inheriting from `biooncograph.data.Dataset`. You must implement the `process()` method to parse your raw data and construct the `OncoGraph` object. See the [Tutorials](tutorials.md) for a step-by-step guide.

**Q: Does BioOncoGraph support dynamic or temporal graphs?**
**A:** Basic support is provided in Track 12 (Resistance Mechanism Prediction). Full dynamic graph neural network architectures are currently in beta and will be fully released in v2.1.

**Q: How can I extract node embeddings from a trained model?**
**A:** You can run a forward pass through the encoder only:
```python
model.eval()
with torch.no_grad():
    node_embeddings = model.encoder(graph.x_dict, graph.edge_index_dict)
    # node_embeddings is a dictionary mapping node types to embedding tensors
```
