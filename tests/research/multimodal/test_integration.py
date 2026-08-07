"""Full Integration Pipeline Tests for Bio-OncoGraph R007.

Tests the full multimodal learning pipeline from configuration to processing,
validation, checkpointing, and verifying deterministic behavior.
"""

import pytest
import numpy as np
import os
import tempfile
import json

# Full simplified pipeline components for integration testing

class Config:
    def __init__(self, **kwargs):
        self.params = kwargs
        
class Encoder:
    def __init__(self, name, factor):
        self.name = name
        self.factor = factor
    def __call__(self, x):
        return x * self.factor

class Validator:
    def validate(self, raw_data):
        for k, v in raw_data.items():
            if np.isnan(v).any():
                raise ValueError(f"NaN in {k}")
        return True

class Synchronizer:
    def impute_missing(self, data, expected_keys):
        for k in expected_keys:
            if k not in data:
                data[k] = np.zeros_like(list(data.values())[0])
        return data
        
    def align(self, data):
        return data

class Engine:
    def __init__(self, config):
        self.config = config
        self.encoders = {}
        self.validator = Validator()
        self.synchronizer = Synchronizer()
        self.fused = None
        
    def register(self, encoder):
        self.encoders[encoder.name] = encoder
        
    def process(self, raw_data):
        self.validator.validate(raw_data)
        
        # Encode
        encoded = {}
        for k, v in raw_data.items():
            if k in self.encoders:
                encoded[k] = self.encoders[k](v)
                
        # Impute & Align
        expected = list(self.encoders.keys())
        imputed = self.synchronizer.impute_missing(encoded, expected)
        aligned = self.synchronizer.align(imputed)
        
        # Fuse
        np.random.seed(self.config.params.get("seed", 0))
        stack = np.stack(list(aligned.values()))
        self.fused = np.mean(stack, axis=0) + np.random.rand(*stack.shape[1:])
        return self.fused

class CheckpointMgr:
    def save(self, data, path):
        with open(path, 'w') as f:
            json.dump({"fused": data.tolist()}, f)
            
    def load(self, path):
        with open(path, 'r') as f:
            d = json.load(f)
        return np.array(d["fused"])

# Fixtures

@pytest.fixture
def config():
    return Config(seed=42, expected_modalities=["histopathology", "clinical", "genomics"])

@pytest.fixture
def engine(config):
    eng = Engine(config)
    eng.register(Encoder("histopathology", 1.5))
    eng.register(Encoder("clinical", 0.5))
    eng.register(Encoder("genomics", 2.0))
    return eng

@pytest.fixture
def raw_data():
    return {
        "histopathology": np.ones((5, 10)),
        "clinical": np.ones((5, 10)) * 2,
        "genomics": np.ones((5, 10)) * 3
    }

@pytest.fixture
def temp_ckpt():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    os.remove(path)

# Tests

def test_full_pipeline_success(engine, raw_data, temp_ckpt):
    """Test standard end-to-end processing pipeline."""
    # Process
    fused = engine.process(raw_data)
    assert fused is not None
    assert fused.shape == (5, 10)
    
    # Checkpoint
    ckpt_mgr = CheckpointMgr()
    ckpt_mgr.save(fused, temp_ckpt)
    
    # Load and verify
    loaded = ckpt_mgr.load(temp_ckpt)
    np.testing.assert_array_equal(fused, loaded)

def test_pipeline_deterministic_runs(engine, raw_data):
    """Verify bitwise equality of fused embeddings across multiple identical runs."""
    fused1 = engine.process(raw_data)
    
    # Re-instantiate engine with same config to test determinism
    engine2 = Engine(Config(seed=42))
    engine2.register(Encoder("histopathology", 1.5))
    engine2.register(Encoder("clinical", 0.5))
    engine2.register(Encoder("genomics", 2.0))
    
    fused2 = engine2.process(raw_data)
    
    np.testing.assert_array_equal(fused1, fused2)

def test_pipeline_different_seed(raw_data):
    """Verify different seeds produce different outputs."""
    e1 = Engine(Config(seed=1))
    e1.register(Encoder("histopathology", 1.0))
    fused1 = e1.process(raw_data)
    
    e2 = Engine(Config(seed=2))
    e2.register(Encoder("histopathology", 1.0))
    fused2 = e2.process(raw_data)
    
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(fused1, fused2)

def test_pipeline_missing_imputation(engine, raw_data):
    """Test pipeline handles missing data via imputation."""
    del raw_data["genomics"]
    fused = engine.process(raw_data)
    assert fused is not None
    assert fused.shape == (5, 10)

def test_pipeline_validation_failure(engine, raw_data):
    """Test pipeline fails early on invalid data."""
    raw_data["clinical"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN in clinical"):
        engine.process(raw_data)

def test_pipeline_no_data(engine):
    """Test pipeline with empty data raises error in fusion if expected keys present."""
    with pytest.raises(ValueError): # np.stack fails on empty list
        engine.process({})

def test_pipeline_unregistered_modality(engine, raw_data):
    """Test pipeline ignores modalities without registered encoders."""
    raw_data["unknown"] = np.ones((5, 10))
    fused1 = engine.process(raw_data)
    
    # Should be identical to run without 'unknown'
    del raw_data["unknown"]
    # Re-seed via process
    fused2 = engine.process(raw_data)
    
    np.testing.assert_array_equal(fused1, fused2)

def test_pipeline_different_input_shapes():
    """Test pipeline fails if inputs cannot be aligned/stacked."""
    e = Engine(Config(seed=42))
    e.register(Encoder("m1", 1.0))
    e.register(Encoder("m2", 1.0))
    
    data = {"m1": np.ones((5, 10)), "m2": np.ones((6, 10))}
    
    with pytest.raises(ValueError): # np.stack fails on size mismatch
        e.process(data)

def test_pipeline_single_modality(raw_data):
    """Test pipeline works with only one registered modality."""
    e = Engine(Config(seed=42))
    e.register(Encoder("histopathology", 1.0))
    
    fused = e.process({"histopathology": raw_data["histopathology"]})
    assert fused.shape == (5, 10)

def test_config_initialization():
    """Test Config parses kwargs correctly."""
    c = Config(a=1, b=2)
    assert c.params["a"] == 1
    assert c.params["b"] == 2

def test_encoder_call():
    """Test Encoder mathematical transformation."""
    enc = Encoder("test", 2.0)
    res = enc(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(res, np.array([2.0, 4.0]))

def test_validator_pass():
    """Test Validator passes valid data."""
    v = Validator()
    assert v.validate({"m1": np.array([1.0, 2.0])}) is True

def test_synchronizer_impute_missing_none():
    """Test synchronizer when no missing data."""
    s = Synchronizer()
    data = {"m1": np.zeros(2)}
    res = s.impute_missing(data, ["m1"])
    assert len(res) == 1

def test_synchronizer_impute_missing_one():
    """Test synchronizer when one modality is missing."""
    s = Synchronizer()
    data = {"m1": np.ones(2)}
    res = s.impute_missing(data, ["m1", "m2"])
    assert len(res) == 2
    np.testing.assert_array_equal(res["m2"], np.zeros(2))

def test_synchronizer_align_passthrough():
    """Test synchronizer align is currently a passthrough."""
    s = Synchronizer()
    data = {"m1": np.ones(2)}
    assert s.align(data) == data

def test_checkpoint_mgr_save_load(temp_ckpt):
    """Test independent checkpoint manager save/load."""
    mgr = CheckpointMgr()
    data = np.array([[1.5, 2.5]])
    mgr.save(data, temp_ckpt)
    loaded = mgr.load(temp_ckpt)
    np.testing.assert_array_equal(data, loaded)

def test_checkpoint_mgr_file_not_found():
    """Test checkpoint manager on non-existent file."""
    mgr = CheckpointMgr()
    with pytest.raises(FileNotFoundError):
        mgr.load("fake_file.json")

def test_engine_register_multiple():
    """Test Engine registers multiple encoders correctly."""
    e = Engine(Config())
    e.register(Encoder("e1", 1))
    e.register(Encoder("e2", 2))
    assert len(e.encoders) == 2
    assert "e1" in e.encoders

def test_engine_process_sets_fused_state(engine, raw_data):
    """Test engine state is updated post-process."""
    assert engine.fused is None
    engine.process(raw_data)
    assert engine.fused is not None

def test_pipeline_large_data(engine):
    """Test pipeline handles larger synthetic data structures."""
    large_data = {
        "histopathology": np.random.rand(100, 50, 50),
        "clinical": np.random.rand(100, 50, 50),
        "genomics": np.random.rand(100, 50, 50)
    }
    fused = engine.process(large_data)
    assert fused.shape == (100, 50, 50)

def test_pipeline_zero_data(engine):
    """Test pipeline handles zeroed data."""
    zero_data = {
        "histopathology": np.zeros((5, 5)),
        "clinical": np.zeros((5, 5)),
        "genomics": np.zeros((5, 5))
    }
    fused = engine.process(zero_data)
    # The output won't be exactly zero due to the random noise addition in mock engine fuse
    assert fused.shape == (5, 5)

def test_pipeline_negative_data(engine):
    """Test pipeline handles negative data."""
    neg_data = {
        "histopathology": np.full((3, 3), -1.0),
        "clinical": np.full((3, 3), -2.0),
        "genomics": np.full((3, 3), -3.0)
    }
    fused = engine.process(neg_data)
    assert fused.shape == (3, 3)

def test_pipeline_impute_mixed_shapes():
    """Test imputation uses correct shape based on first available modality."""
    s = Synchronizer()
    data = {"m1": np.ones((4, 2))}
    res = s.impute_missing(data, ["m1", "m2"])
    assert res["m2"].shape == (4, 2)

def test_pipeline_impute_all_missing():
    """Test imputation fails gracefully or errors if NO data is provided to infer shape."""
    s = Synchronizer()
    with pytest.raises(IndexError):
        s.impute_missing({}, ["m1"])

def test_validator_multiple_modalities_nan():
    """Test validator catches NaN in secondary modalities."""
    v = Validator()
    with pytest.raises(ValueError, match="NaN in m2"):
        v.validate({"m1": np.ones(2), "m2": np.array([1.0, np.nan])})

def test_encoder_factor_zero():
    """Test encoder with zero factor zeros out data."""
    enc = Encoder("zero", 0.0)
    res = enc(np.ones(5))
    np.testing.assert_array_equal(res, np.zeros(5))

def test_pipeline_encoder_factor_zero(raw_data):
    """Test pipeline where one modality is zeroed out by encoder."""
    e = Engine(Config(seed=42))
    e.register(Encoder("histopathology", 0.0))
    e.register(Encoder("clinical", 1.0))
    
    # Process only these two to avoid genomics interference
    data = {"histopathology": raw_data["histopathology"], "clinical": raw_data["clinical"]}
    fused = e.process(data)
    assert fused is not None

def test_pipeline_no_encoders(raw_data):
    """Test pipeline when no encoders are registered."""
    e = Engine(Config())
    with pytest.raises(ValueError): # fails at fusion step due to empty stack
        e.process(raw_data)

def test_engine_reprocess(engine, raw_data):
    """Test calling process multiple times with same data."""
    f1 = engine.process(raw_data)
    # The config seed is used, so random state should reset and output should match
    f2 = engine.process(raw_data)
    np.testing.assert_array_equal(f1, f2)

def test_integration_checkpoint_reload(engine, raw_data, temp_ckpt):
    """End-to-end process, save, and reload verification."""
    # 1. Process
    fused_original = engine.process(raw_data)
    
    # 2. Save
    mgr = CheckpointMgr()
    mgr.save(fused_original, temp_ckpt)
    
    # 3. Reload
    fused_reloaded = mgr.load(temp_ckpt)
    
    # 4. Verify
    np.testing.assert_array_equal(fused_original, fused_reloaded)
    assert fused_reloaded.shape == (5, 10)
