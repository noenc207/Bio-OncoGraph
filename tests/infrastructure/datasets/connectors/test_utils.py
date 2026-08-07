"""Tests for utility functions in Dataset Connectors."""

import pytest
import re
from typing import Dict, Any, List

def sanitize_name(name: str) -> str:
    """Convert dataset name to a standard format (lowercase, no spaces)."""
    if not isinstance(name, str):
        raise TypeError("Name must be a string")
    return re.sub(r'[^a-z0-9_]', '_', name.lower().strip())

def parse_config(config_str: str) -> Dict[str, str]:
    """Parse a basic key=value configuration string."""
    config = {}
    for line in config_str.split('\n'):
        line = line.strip()
        if line and '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            config[k.strip()] = v.strip()
    return config

def chunk_list(data: List[Any], size: int) -> List[List[Any]]:
    """Split a list into smaller chunks of given size."""
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    return [data[i:i + size] for i in range(0, len(data), size)]

def test_sanitize_name_normal():
    assert sanitize_name("My Dataset") == "my_dataset"

def test_sanitize_name_special_chars():
    assert sanitize_name("Data-Set@123!") == "data_set_123_"

def test_sanitize_name_empty():
    assert sanitize_name("") == ""

def test_sanitize_name_already_clean():
    assert sanitize_name("my_dataset_1") == "my_dataset_1"

def test_sanitize_name_invalid_type():
    with pytest.raises(TypeError):
        sanitize_name(123)

def test_parse_config_normal():
    config_str = "host=localhost\nport=8080"
    assert parse_config(config_str) == {"host": "localhost", "port": "8080"}

def test_parse_config_with_comments():
    config_str = "# Database\nhost=db\n# Port\nport=5432"
    assert parse_config(config_str) == {"host": "db", "port": "5432"}

def test_parse_config_empty():
    assert parse_config("") == {}

def test_parse_config_malformed():
    config_str = "host=localhost\nbad_line"
    assert parse_config(config_str) == {"host": "localhost"}

def test_chunk_list_normal():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

def test_chunk_list_exact_multiple():
    assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

def test_chunk_list_empty():
    assert chunk_list([], 3) == []

def test_chunk_list_invalid_size():
    with pytest.raises(ValueError):
        chunk_list([1, 2], 0)

def test_chunk_list_large_size():
    assert chunk_list([1, 2], 10) == [[1, 2]]
