"""Tests for Utils.

This module tests sanitize_module_name, generate_timestamp, compute_content_hash, validate_semver.
"""

import pytest
import re
import hashlib
from datetime import datetime

# Utility functions
def sanitize_module_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()

def generate_timestamp() -> str:
    return datetime.utcnow().isoformat()

def compute_content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def validate_semver(version: str) -> bool:
    pattern = r'^\d+\.\d+\.\d+(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$'
    return bool(re.match(pattern, version))

@pytest.mark.parametrize("input_name, expected", [
    ("MyModule", "mymodule"),
    ("My-Module", "my_module"),
    ("Test 123", "test_123"),
    ("Special!@#Chars", "special___chars"),
    ("ALREADY_CLEAN", "already_clean")
])
@pytest.mark.parametrize("prefix", ["", "prefix_"])
def test_sanitize_module_name(input_name: str, expected: str, prefix: str) -> None:
    """Test sanitizing module names."""
    assert sanitize_module_name(prefix + input_name) == (prefix + expected).lower()

@pytest.mark.parametrize("runs", range(5))
def test_generate_timestamp(runs: int) -> None:
    """Test generating timestamps."""
    ts = generate_timestamp()
    assert isinstance(ts, str)
    assert "T" in ts

@pytest.mark.parametrize("content", [
    b"test content",
    b"another content",
    b"",
    b"12345",
    b"\x00\x01\x02"
])
@pytest.mark.parametrize("salt", [b"", b"salt1", b"salt2"])
def test_compute_content_hash(content: bytes, salt: bytes) -> None:
    """Test computing content hash."""
    hashed = compute_content_hash(content + salt)
    assert len(hashed) == 64
    assert isinstance(hashed, str)

@pytest.mark.parametrize("version, valid", [
    ("1.0.0", True),
    ("0.1.0-alpha", True),
    ("2.0.0-beta.1", True),
    ("1.0.0+build.123", True),
    ("1.0", False),
    ("v1.0.0", False),
    ("1.0.0.0", False),
    ("invalid", False)
])
@pytest.mark.parametrize("suffix", ["", " "])
def test_validate_semver(version: str, valid: bool, suffix: str) -> None:
    """Test validating semantic versioning."""
    if suffix:
        assert validate_semver(version + suffix) is False
    else:
        assert validate_semver(version) is valid
