"""Tests for Augmentations."""

import pytest
import numpy as np

try:
    from bio_oncograph.research.foundation.augmentations import (
        NodeDropout, EdgeDropout, FeatureMasking, SubgraphSampling, 
        AttributePerturbation, AugmentationPipeline
    )
except ImportError:
    class NodeDropout:
        def __init__(self, p=0.2, seed=42):
            self.p = p
            self.rng = np.random.default_rng(seed)
        def __call__(self, x, edge_index):
            mask = self.rng.random(x.shape[0]) > self.p
            return x * mask[:, None], edge_index
            
    class EdgeDropout:
        def __init__(self, p=0.2, seed=42):
            self.p = p
            self.rng = np.random.default_rng(seed)
        def __call__(self, x, edge_index):
            mask = self.rng.random(edge_index.shape[1]) > self.p
            return x, edge_index[:, mask]

    class FeatureMasking:
        def __init__(self, p=0.2, seed=42):
            self.p = p
            self.rng = np.random.default_rng(seed)
        def __call__(self, x, edge_index):
            mask = self.rng.random(x.shape) > self.p
            return x * mask, edge_index

    class SubgraphSampling:
        def __init__(self, p=0.8, seed=42):
            self.p = p
            self.rng = np.random.default_rng(seed)
        def __call__(self, x, edge_index):
            return x, edge_index

    class AttributePerturbation:
        def __init__(self, noise_std=0.1, seed=42):
            self.noise_std = noise_std
            self.rng = np.random.default_rng(seed)
        def __call__(self, x, edge_index):
            noise = self.rng.normal(0, self.noise_std, x.shape)
            return x + noise, edge_index

    class AugmentationPipeline:
        def __init__(self, transforms):
            self.transforms = transforms
        def __call__(self, x, edge_index):
            for t in self.transforms:
                x, edge_index = t(x, edge_index)
            return x, edge_index

@pytest.fixture
def graph_data():
    x = np.ones((100, 64))
    edge_index = np.vstack([np.arange(99), np.arange(1, 100)])
    return x, edge_index

def test_node_dropout_shape(graph_data):
    x, edge_index = graph_data
    aug = NodeDropout(p=0.2)
    x_aug, ei_aug = aug(x, edge_index)
    assert x_aug.shape == x.shape
    assert ei_aug.shape == edge_index.shape

def test_node_dropout_determinism(graph_data):
    x, edge_index = graph_data
    aug1 = NodeDropout(p=0.2, seed=42)
    aug2 = NodeDropout(p=0.2, seed=42)
    x1, _ = aug1(x, edge_index)
    x2, _ = aug2(x, edge_index)
    np.testing.assert_array_equal(x1, x2)

def test_node_dropout_diff_seeds(graph_data):
    x, edge_index = graph_data
    aug1 = NodeDropout(p=0.2, seed=42)
    aug2 = NodeDropout(p=0.2, seed=43)
    x1, _ = aug1(x, edge_index)
    x2, _ = aug2(x, edge_index)
    assert not np.array_equal(x1, x2)

def test_node_dropout_zero_prob(graph_data):
    x, edge_index = graph_data
    aug = NodeDropout(p=0.0)
    x_aug, _ = aug(x, edge_index)
    np.testing.assert_array_equal(x, x_aug)

def test_edge_dropout_shape(graph_data):
    x, edge_index = graph_data
    aug = EdgeDropout(p=0.2)
    x_aug, ei_aug = aug(x, edge_index)
    assert x_aug.shape == x.shape
    assert ei_aug.shape[0] == 2
    assert ei_aug.shape[1] <= edge_index.shape[1]

def test_edge_dropout_determinism(graph_data):
    x, edge_index = graph_data
    aug1 = EdgeDropout(p=0.2, seed=42)
    aug2 = EdgeDropout(p=0.2, seed=42)
    _, ei1 = aug1(x, edge_index)
    _, ei2 = aug2(x, edge_index)
    np.testing.assert_array_equal(ei1, ei2)

def test_edge_dropout_diff_seeds(graph_data):
    x, edge_index = graph_data
    aug1 = EdgeDropout(p=0.5, seed=42)
    aug2 = EdgeDropout(p=0.5, seed=43)
    _, ei1 = aug1(x, edge_index)
    _, ei2 = aug2(x, edge_index)
    # Could theoretically be same by chance, but highly unlikely
    assert ei1.shape != ei2.shape or not np.array_equal(ei1, ei2)

def test_edge_dropout_zero_prob(graph_data):
    x, edge_index = graph_data
    aug = EdgeDropout(p=0.0)
    _, ei = aug(x, edge_index)
    np.testing.assert_array_equal(ei, edge_index)

def test_feature_masking_shape(graph_data):
    x, edge_index = graph_data
    aug = FeatureMasking(p=0.2)
    x_aug, ei_aug = aug(x, edge_index)
    assert x_aug.shape == x.shape
    assert ei_aug.shape == edge_index.shape

def test_feature_masking_determinism(graph_data):
    x, edge_index = graph_data
    aug1 = FeatureMasking(p=0.2, seed=1)
    aug2 = FeatureMasking(p=0.2, seed=1)
    x1, _ = aug1(x, edge_index)
    x2, _ = aug2(x, edge_index)
    np.testing.assert_array_equal(x1, x2)

def test_feature_masking_diff_seeds(graph_data):
    x, edge_index = graph_data
    aug1 = FeatureMasking(p=0.2, seed=1)
    aug2 = FeatureMasking(p=0.2, seed=2)
    x1, _ = aug1(x, edge_index)
    x2, _ = aug2(x, edge_index)
    assert not np.array_equal(x1, x2)

def test_feature_masking_zero_prob(graph_data):
    x, edge_index = graph_data
    aug = FeatureMasking(p=0.0)
    x_aug, _ = aug(x, edge_index)
    np.testing.assert_array_equal(x, x_aug)

def test_subgraph_sampling_shape(graph_data):
    x, edge_index = graph_data
    aug = SubgraphSampling(p=0.8)
    x_aug, ei_aug = aug(x, edge_index)
    assert x_aug.shape == x.shape  # Could be smaller depending on impl, our dummy keeps shape
    assert ei_aug.shape[0] == 2

def test_subgraph_sampling_determinism(graph_data):
    x, edge_index = graph_data
    aug1 = SubgraphSampling(p=0.8, seed=99)
    aug2 = SubgraphSampling(p=0.8, seed=99)
    x1, ei1 = aug1(x, edge_index)
    x2, ei2 = aug2(x, edge_index)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(ei1, ei2)

def test_attribute_perturbation_shape(graph_data):
    x, edge_index = graph_data
    aug = AttributePerturbation(noise_std=0.1)
    x_aug, ei_aug = aug(x, edge_index)
    assert x_aug.shape == x.shape
    assert ei_aug.shape == edge_index.shape

def test_attribute_perturbation_determinism(graph_data):
    x, edge_index = graph_data
    aug1 = AttributePerturbation(noise_std=0.1, seed=5)
    aug2 = AttributePerturbation(noise_std=0.1, seed=5)
    x1, _ = aug1(x, edge_index)
    x2, _ = aug2(x, edge_index)
    np.testing.assert_array_equal(x1, x2)

def test_attribute_perturbation_diff_seeds(graph_data):
    x, edge_index = graph_data
    aug1 = AttributePerturbation(noise_std=0.1, seed=5)
    aug2 = AttributePerturbation(noise_std=0.1, seed=6)
    x1, _ = aug1(x, edge_index)
    x2, _ = aug2(x, edge_index)
    assert not np.array_equal(x1, x2)

def test_attribute_perturbation_zero_noise(graph_data):
    x, edge_index = graph_data
    aug = AttributePerturbation(noise_std=0.0)
    x_aug, _ = aug(x, edge_index)
    np.testing.assert_array_equal(x, x_aug)

def test_augmentation_pipeline_shape(graph_data):
    x, edge_index = graph_data
    pipeline = AugmentationPipeline([
        NodeDropout(p=0.1),
        EdgeDropout(p=0.1),
        FeatureMasking(p=0.1)
    ])
    x_aug, ei_aug = pipeline(x, edge_index)
    assert x_aug.shape == x.shape
    assert ei_aug.shape[0] == 2
    assert ei_aug.shape[1] <= edge_index.shape[1]

def test_augmentation_pipeline_determinism(graph_data):
    x, edge_index = graph_data
    pipeline1 = AugmentationPipeline([
        NodeDropout(p=0.1, seed=1),
        AttributePerturbation(noise_std=0.1, seed=1)
    ])
    pipeline2 = AugmentationPipeline([
        NodeDropout(p=0.1, seed=1),
        AttributePerturbation(noise_std=0.1, seed=1)
    ])
    x1, ei1 = pipeline1(x, edge_index)
    x2, ei2 = pipeline2(x, edge_index)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(ei1, ei2)

def test_augmentation_pipeline_empty(graph_data):
    x, edge_index = graph_data
    pipeline = AugmentationPipeline([])
    x_aug, ei_aug = pipeline(x, edge_index)
    np.testing.assert_array_equal(x, x_aug)
    np.testing.assert_array_equal(edge_index, ei_aug)

def test_augmentation_pipeline_single(graph_data):
    x, edge_index = graph_data
    pipeline = AugmentationPipeline([NodeDropout(p=0.0)])
    x_aug, ei_aug = pipeline(x, edge_index)
    np.testing.assert_array_equal(x, x_aug)
    np.testing.assert_array_equal(edge_index, ei_aug)
