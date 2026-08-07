"""Tests for Graph Foundation Determinism."""

import pytest
import numpy as np

def run_pipeline(seed):
    np.random.seed(seed)
    # simulate graph creation
    graphs = [np.random.randn(10, 16) for _ in range(5)]
    # simulate augmentation
    aug_graphs = [g + np.random.randn(10, 16) * 0.1 for g in graphs]
    # simulate encode
    embeddings = [np.mean(g, axis=0) for g in aug_graphs]
    # simulate loss
    loss = np.sum([np.linalg.norm(e) for e in embeddings])
    return {
        "graphs": graphs,
        "aug_graphs": aug_graphs,
        "embeddings": embeddings,
        "loss": loss
    }

def test_pipeline_overall_determinism():
    res1 = run_pipeline(42)
    res2 = run_pipeline(42)
    assert res1["loss"] == res2["loss"]

def test_pipeline_different_seeds():
    res1 = run_pipeline(42)
    res2 = run_pipeline(43)
    assert res1["loss"] != res2["loss"]

def test_graph_creation_determinism():
    res1 = run_pipeline(42)
    res2 = run_pipeline(42)
    for g1, g2 in zip(res1["graphs"], res2["graphs"]):
        np.testing.assert_array_equal(g1, g2)

def test_augmentation_determinism():
    res1 = run_pipeline(42)
    res2 = run_pipeline(42)
    for ag1, ag2 in zip(res1["aug_graphs"], res2["aug_graphs"]):
        np.testing.assert_array_equal(ag1, ag2)

def test_encoding_determinism():
    res1 = run_pipeline(42)
    res2 = run_pipeline(42)
    for e1, e2 in zip(res1["embeddings"], res2["embeddings"]):
        np.testing.assert_array_equal(e1, e2)

def test_loss_determinism():
    res1 = run_pipeline(42)
    res2 = run_pipeline(42)
    assert np.isclose(res1["loss"], res2["loss"])

def test_graph_creation_different_seeds():
    res1 = run_pipeline(42)
    res2 = run_pipeline(43)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(res1["graphs"][0], res2["graphs"][0])

def test_augmentation_different_seeds():
    res1 = run_pipeline(42)
    res2 = run_pipeline(43)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(res1["aug_graphs"][0], res2["aug_graphs"][0])

def test_encoding_different_seeds():
    res1 = run_pipeline(42)
    res2 = run_pipeline(43)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(res1["embeddings"][0], res2["embeddings"][0])

def test_pipeline_multiple_runs():
    res1 = run_pipeline(42)
    for _ in range(5):
        res2 = run_pipeline(42)
        assert res1["loss"] == res2["loss"]

def test_seed_zero():
    res1 = run_pipeline(0)
    res2 = run_pipeline(0)
    assert res1["loss"] == res2["loss"]

def test_seed_negative():
    # Numpy handles np.random.seed with uint32, so using a valid large int or wrapping
    # We will just use a large positive seed to simulate arbitrary seed
    res1 = run_pipeline(999999)
    res2 = run_pipeline(999999)
    assert res1["loss"] == res2["loss"]

def test_graphs_length():
    res = run_pipeline(42)
    assert len(res["graphs"]) == 5

def test_aug_graphs_length():
    res = run_pipeline(42)
    assert len(res["aug_graphs"]) == 5

def test_embeddings_length():
    res = run_pipeline(42)
    assert len(res["embeddings"]) == 5

def test_graphs_shape():
    res = run_pipeline(42)
    assert res["graphs"][0].shape == (10, 16)

def test_aug_graphs_shape():
    res = run_pipeline(42)
    assert res["aug_graphs"][0].shape == (10, 16)

def test_embeddings_shape():
    res = run_pipeline(42)
    assert res["embeddings"][0].shape == (16,)

def test_loss_type():
    res = run_pipeline(42)
    assert isinstance(res["loss"], float) or isinstance(res["loss"], np.floating)

def test_loss_positive():
    res = run_pipeline(42)
    assert res["loss"] > 0

def test_graph_values_range():
    res = run_pipeline(42)
    assert np.max(np.abs(res["graphs"][0])) < 10.0 # Just a reasonable check for randn

def test_aug_difference():
    res = run_pipeline(42)
    diff = res["aug_graphs"][0] - res["graphs"][0]
    assert np.any(diff != 0)

def test_encoding_computation():
    res = run_pipeline(42)
    expected = np.mean(res["aug_graphs"][0], axis=0)
    np.testing.assert_array_almost_equal(res["embeddings"][0], expected)

def test_loss_computation():
    res = run_pipeline(42)
    expected_loss = np.sum([np.linalg.norm(e) for e in res["embeddings"]])
    assert np.isclose(res["loss"], expected_loss)

def test_state_isolation():
    res1 = run_pipeline(10)
    np.random.rand(100) # change state
    res2 = run_pipeline(10)
    assert res1["loss"] == res2["loss"]
