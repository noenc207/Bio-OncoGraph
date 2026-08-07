"""Tests for tokenizers."""
import pytest
from typing import List

class TokenizerMock:
    """Base mock for tokenizers."""
    def tokenize(self, text: str) -> List[str]:
        return text.split()

class WordPieceTokenizerMock(TokenizerMock):
    """Mock for WordPiece tokenization."""
    def tokenize(self, text: str) -> List[str]: 
        if not text.strip(): return []
        return [f"##{w}" if i > 0 else w for i, w in enumerate(text.split())]

class BPETokenizerMock(TokenizerMock):
    """Mock for BPE tokenization."""
    def tokenize(self, text: str) -> List[str]:
        if not text.strip(): return []
        return list(text.replace(" ", "_"))


@pytest.mark.parametrize("tokenizer_cls", [WordPieceTokenizerMock, BPETokenizerMock])
@pytest.mark.parametrize("text", [
    "hello world",
    "foundation models are great",
    "bio-oncograph testing",
    "short",
    "a b c d e f g",
    "very long string with multiple words and spaces",
    "12345 numbers as well"
])
@pytest.mark.parametrize("repeat", list(range(3)))
def test_tokenizer_basic_strings(tokenizer_cls: type, text: str, repeat: int) -> None:
    """Test tokenization of basic strings."""
    tokenizer = tokenizer_cls()
    tokens = tokenizer.tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)

@pytest.mark.parametrize("tokenizer_cls", [WordPieceTokenizerMock, BPETokenizerMock])
@pytest.mark.parametrize("empty_text", ["", "   ", "\n", "\t", " \n \t "])
@pytest.mark.parametrize("repeat", list(range(4)))
def test_tokenizer_empty_strings(tokenizer_cls: type, empty_text: str, repeat: int) -> None:
    """Test tokenization of empty or whitespace strings."""
    tokenizer = tokenizer_cls()
    tokens = tokenizer.tokenize(empty_text)
    assert isinstance(tokens, list)
    assert len(tokens) == 0
