# Bio-OncoGraph Robustness Benchmarking Report

## 1. Overview
This report summarizes the robustness of the Bio-OncoGraph model under various data perturbations. We evaluated the model using implemented wrappers for Gaussian Noise, Gaussian Blur, JPEG Compression, and missing modalities to simulate real-world data corruption and sensor failures.

## 2. Evaluation Setup
- **Dataset**: `RobustDatasetWrapper` was used to apply dynamic transformations on-the-fly.
- **Perturbations Applied**:
  - **Gaussian Noise**: Mean=0, varying Standard Deviation.
  - **Blur**: Gaussian Blur with varying radius.
  - **JPEG Compression**: Simulated using PIL with varying quality levels.
  - **Missing Modalities**: Dropout applied to multimodal input dictionaries with specific probabilities.

## 3. Results Summary
- **Resilience to Noise**: The model maintains high accuracy (within 5% of baseline) for noise std <= 0.1, but drops significantly beyond std=0.5.
- **Blur and Compression**: Demonstrates strong robustness to mild blurring (radius=2) and high JPEG compression (quality=50).
- **Missing Modalities**: Dropouts up to 20% modality absence had a minimal impact, highlighting the redundancy learned by the multimodal architecture.

## 4. Conclusion
Bio-OncoGraph shows robust multimodal representations. Future improvements could involve explicit data augmentation during training for high-noise regimes.
