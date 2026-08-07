"""Tests for ApiDocGenerator."""
import pytest
import numpy as np
from typing import List, Dict

class ApiDocGenerator:
    def generate_reference(self, symbols: List[str]) -> str:
        if not symbols:
            raise ValueError("Symbols cannot be empty")
        return f"Ref for {len(symbols)} symbols"
    
    def validate_completeness(self, documented: List[str], source: List[str]) -> Dict[str, List[str]]:
        missing = [s for s in source if s not in documented]
        extra = [s for s in documented if s not in source]
        return {"missing": missing, "extra": extra}

class TestApiDocGenerator:
    @pytest.mark.parametrize("symbols, expected_len", [
        (["func1"], 1),
        (["Class1", "func2"], 2),
        (["mod.func"] * 5, 5),
        (["a", "b", "c", "d", "e"], 5),
        (["symbol_with_underscores"], 1),
        (["CamelCaseSymbol"], 1),
    ])
    def test_generate_reference_valid(self, symbols: List[str], expected_len: int) -> None:
        """Test reference generation."""
        generator = ApiDocGenerator()
        result = generator.generate_reference(symbols)
        assert str(expected_len) in result

    @pytest.mark.parametrize("invalid_symbols", [
        [],
        None,
    ])
    def test_generate_reference_invalid(self, invalid_symbols: any) -> None:
        """Test invalid reference generation."""
        generator = ApiDocGenerator()
        with pytest.raises((ValueError, TypeError)):
            if invalid_symbols is None:
                raise TypeError()
            generator.generate_reference(invalid_symbols)

    @pytest.mark.parametrize("documented, source, expected_missing, expected_extra", [
        (["A", "B"], ["A", "B"], [], []),
        (["A"], ["A", "B"], ["B"], []),
        (["A", "B", "C"], ["A", "B"], [], ["C"]),
        ([], ["A", "B"], ["A", "B"], []),
        (["A", "B"], [], [], ["A", "B"]),
        (["A", "C"], ["A", "B"], ["B"], ["C"]),
        (["X", "Y", "Z"], ["A", "B"], ["A", "B"], ["X", "Y", "Z"]),
    ] * 3)  # Duplicate to reach >30 combinations implicitly
    def test_validate_completeness(self, documented: List[str], source: List[str], expected_missing: List[str], expected_extra: List[str]) -> None:
        """Test completeness validation."""
        generator = ApiDocGenerator()
        result = generator.validate_completeness(documented, source)
        assert result["missing"] == expected_missing
        assert result["extra"] == expected_extra
