"""Tests for trust aggregation."""

import numpy as np
import pytest

class TrustAggregator:
    def __init__(self, weight_confidence=0.5, weight_consistency=0.5):
        self.w_conf = weight_confidence
        self.w_cons = weight_consistency

    def compute_trust_score(self, confidence, consistency):
        score = self.w_conf * confidence + self.w_cons * consistency
        return np.clip(score, 0.0, 1.0)

def test_trust_aggregator_perfect():
    agg = TrustAggregator()
    score = agg.compute_trust_score(1.0, 1.0)
    assert score == 1.0

def test_trust_aggregator_zero():
    agg = TrustAggregator()
    score = agg.compute_trust_score(0.0, 0.0)
    assert score == 0.0

def test_trust_aggregator_mid():
    agg = TrustAggregator()
    score = agg.compute_trust_score(0.5, 0.5)
    assert score == 0.5

def test_trust_aggregator_weights():
    agg = TrustAggregator(weight_confidence=0.8, weight_consistency=0.2)
    score = agg.compute_trust_score(1.0, 0.0)
    assert np.isclose(score, 0.8)

def test_trust_aggregator_clip_high():
    agg = TrustAggregator(weight_confidence=1.0, weight_consistency=1.0)
    score = agg.compute_trust_score(1.0, 1.0)
    assert score == 1.0

def test_trust_aggregator_clip_low():
    agg = TrustAggregator()
    score = agg.compute_trust_score(-1.0, -0.5)
    assert score == 0.0

def test_trust_aggregator_batch():
    agg = TrustAggregator()
    conf = np.array([0.1, 0.9, 0.5])
    cons = np.array([0.9, 0.1, 0.5])
    score = agg.compute_trust_score(conf, cons)
    assert np.allclose(score, [0.5, 0.5, 0.5])

def test_trust_aggregator_stability():
    agg = TrustAggregator()
    score = agg.compute_trust_score(np.array([1.0]), np.array([1.0]))
    assert not np.any(np.isnan(score))
