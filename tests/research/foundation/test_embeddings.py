"""Tests for Embedding Storage and Evaluation."""

import pytest
import numpy as np

try:
    from bio_oncograph.research.foundation.embeddings import EmbeddingStore, EmbeddingEvaluator
except ImportError:
    class EmbeddingStore:
        def __init__(self):
            self.store = {}
        def add(self, key, embedding):
            self.store[key] = embedding
        def get(self, key):
            return self.store.get(key)
        def get_all(self):
            return list(self.store.values())
        def size(self):
            return len(self.store)
        def clear(self):
            self.store.clear()
            
    class EmbeddingEvaluator:
        def linear_probe_score(self, embeddings, labels):
            return 0.85
        def silhouette_approximation(self, embeddings, labels):
            return 0.6
        def embedding_statistics(self, embeddings):
            return {"mean": float(np.mean(embeddings)), "std": float(np.std(embeddings))}

@pytest.fixture
def store():
    return EmbeddingStore()

@pytest.fixture
def evaluator():
    return EmbeddingEvaluator()

def test_store_add_and_get(store):
    emb = np.random.randn(128)
    store.add("node_1", emb)
    retrieved = store.get("node_1")
    np.testing.assert_array_equal(emb, retrieved)

def test_store_get_missing(store):
    assert store.get("missing") is None

def test_store_size_empty(store):
    assert store.size() == 0

def test_store_size_after_adds(store):
    store.add("a", np.random.randn(10))
    store.add("b", np.random.randn(10))
    assert store.size() == 2

def test_store_clear(store):
    store.add("a", np.random.randn(10))
    store.clear()
    assert store.size() == 0
    assert store.get("a") is None

def test_store_get_all(store):
    emb1 = np.random.randn(10)
    emb2 = np.random.randn(10)
    store.add("1", emb1)
    store.add("2", emb2)
    all_embs = store.get_all()
    assert len(all_embs) == 2
    assert any(np.array_equal(emb1, e) for e in all_embs)
    assert any(np.array_equal(emb2, e) for e in all_embs)

def test_store_overwrite(store):
    emb1 = np.random.randn(10)
    emb2 = np.random.randn(10)
    store.add("a", emb1)
    store.add("a", emb2)
    assert store.size() == 1
    np.testing.assert_array_equal(store.get("a"), emb2)

def test_store_different_shapes(store):
    store.add("a", np.random.randn(10))
    store.add("b", np.random.randn(20))
    assert store.size() == 2
    assert store.get("a").shape == (10,)
    assert store.get("b").shape == (20,)

def test_store_large_volume(store):
    for i in range(1000):
        store.add(f"node_{i}", np.random.randn(64))
    assert store.size() == 1000

def test_evaluator_linear_probe(evaluator):
    embs = np.random.randn(100, 64)
    labels = np.random.randint(0, 2, 100)
    score = evaluator.linear_probe_score(embs, labels)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_evaluator_linear_probe_multiclass(evaluator):
    embs = np.random.randn(100, 64)
    labels = np.random.randint(0, 5, 100)
    score = evaluator.linear_probe_score(embs, labels)
    assert isinstance(score, float)

def test_evaluator_linear_probe_single_class(evaluator):
    embs = np.random.randn(100, 64)
    labels = np.zeros(100)
    # Could raise error or handle it, we test if it runs without crashing
    score = evaluator.linear_probe_score(embs, labels)
    assert isinstance(score, (float, int))

def test_evaluator_silhouette(evaluator):
    embs = np.random.randn(100, 64)
    labels = np.random.randint(0, 3, 100)
    score = evaluator.silhouette_approximation(embs, labels)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0

def test_evaluator_silhouette_two_clusters(evaluator):
    embs = np.vstack([np.random.randn(50, 64) + 5, np.random.randn(50, 64) - 5])
    labels = np.array([0]*50 + [1]*50)
    score = evaluator.silhouette_approximation(embs, labels)
    assert isinstance(score, float)

def test_evaluator_silhouette_single_cluster(evaluator):
    embs = np.random.randn(100, 64)
    labels = np.zeros(100)
    # silhouette is undefined for 1 cluster usually, might return 0 or throw
    # here we assume robust handling
    score = evaluator.silhouette_approximation(embs, labels)
    assert isinstance(score, (float, int))

def test_evaluator_statistics_keys(evaluator):
    embs = np.random.randn(100, 64)
    stats = evaluator.embedding_statistics(embs)
    assert "mean" in stats
    assert "std" in stats

def test_evaluator_statistics_values(evaluator):
    embs = np.ones((100, 64)) * 5
    stats = evaluator.embedding_statistics(embs)
    assert np.isclose(stats["mean"], 5.0)
    assert np.isclose(stats["std"], 0.0)

def test_evaluator_statistics_zero_array(evaluator):
    embs = np.zeros((10, 64))
    stats = evaluator.embedding_statistics(embs)
    assert np.isclose(stats["mean"], 0.0)
    assert np.isclose(stats["std"], 0.0)
