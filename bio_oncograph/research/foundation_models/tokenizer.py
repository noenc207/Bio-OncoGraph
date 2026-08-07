"""Tokenization utilities for foundation models."""

import abc
import logging
from typing import List


logger = logging.getLogger(__name__)


class BaseTokenizer(abc.ABC):
    """Abstract base class for tokenizers."""

    @abc.abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenizes an input string.

        Args:
            text: Input string.

        Returns:
            List of tokens.
        """
        pass

    @abc.abstractmethod
    def detokenize(self, tokens: List[str]) -> str:
        """Detokenizes a list of tokens back to string.

        Args:
            tokens: List of tokens.

        Returns:
            Constructed string.
        """
        pass


class WordPieceTokenizerMock(BaseTokenizer):
    """Mock word piece tokenizer that splits by space."""

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes by spaces."""
        logger.debug("Tokenizing with WordPieceTokenizerMock.")
        return text.split()

    def detokenize(self, tokens: List[str]) -> str:
        """Detokenizes by joining with spaces."""
        logger.debug("Detokenizing with WordPieceTokenizerMock.")
        return " ".join(tokens)


class BPETokenizerMock(BaseTokenizer):
    """Mock BPE tokenizer that splits by character pairs."""

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes by pairing characters."""
        logger.debug("Tokenizing with BPETokenizerMock.")
        tokens = [text[i:i+2] for i in range(0, len(text), 2)]
        if not tokens and text:
            tokens = [text]
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """Detokenizes by concatenation."""
        logger.debug("Detokenizing with BPETokenizerMock.")
        return "".join(tokens)
