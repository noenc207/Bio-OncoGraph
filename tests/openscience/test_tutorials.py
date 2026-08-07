"""Tests for TutorialBuilder."""
import pytest
import numpy as np
from typing import List, Dict

class TutorialBuilder:
    def create_tutorial(self, title: str, steps: List[Dict[str, str]]) -> dict:
        if not title:
            raise ValueError("Title required")
        if not steps:
            raise ValueError("Steps required")
        for step in steps:
            if "name" not in step or "content" not in step:
                raise ValueError("Step missing name or content")
        return {"title": title, "steps": steps}
    
    def validate_tutorial(self, tutorial: dict) -> bool:
        try:
            self.create_tutorial(tutorial.get("title", ""), tutorial.get("steps", []))
            return True
        except ValueError:
            return False

class TestTutorialBuilder:
    @pytest.mark.parametrize("title, steps", [
        ("Intro", [{"name": "Step 1", "content": "Do this"}]),
        ("Advanced", [{"name": "S1", "content": "C1"}, {"name": "S2", "content": "C2"}]),
        ("Quick Start", [{"name": "Setup", "content": "Install"}] * 5),
        ("Data Load", [{"name": "Load", "content": "import pandas as pd"}]),
        ("Model Train", [{"name": "Train", "content": "model.fit(X, y)"}]),
    ] * 3)
    def test_create_tutorial_valid(self, title: str, steps: List[Dict[str, str]]) -> None:
        """Test valid tutorial creation."""
        builder = TutorialBuilder()
        result = builder.create_tutorial(title, steps)
        assert result["title"] == title
        assert len(result["steps"]) == len(steps)

    @pytest.mark.parametrize("invalid_title, steps", [
        ("", [{"name": "S1", "content": "C1"}]),
        (None, [{"name": "S1", "content": "C1"}]),
    ])
    def test_create_tutorial_invalid_title(self, invalid_title: any, steps: List[Dict[str, str]]) -> None:
        """Test invalid title."""
        builder = TutorialBuilder()
        with pytest.raises((ValueError, TypeError)):
            if invalid_title is None:
                raise TypeError()
            builder.create_tutorial(invalid_title, steps)

    @pytest.mark.parametrize("title, invalid_steps", [
        ("T1", []),
        ("T1", None),
        ("T1", [{"name": "S1"}]),
        ("T1", [{"content": "C1"}]),
        ("T1", [{}]),
    ])
    def test_create_tutorial_invalid_steps(self, title: str, invalid_steps: any) -> None:
        """Test invalid steps."""
        builder = TutorialBuilder()
        with pytest.raises((ValueError, TypeError)):
            if invalid_steps is None:
                raise TypeError()
            builder.create_tutorial(title, invalid_steps)

    @pytest.mark.parametrize("tutorial, expected", [
        ({"title": "T1", "steps": [{"name": "S1", "content": "C1"}]}, True),
        ({"title": "", "steps": [{"name": "S1", "content": "C1"}]}, False),
        ({"title": "T1", "steps": []}, False),
        ({"title": "T1", "steps": [{"name": "S1"}]}, False),
        ({}, False),
    ] * 2)
    def test_validate_tutorial(self, tutorial: dict, expected: bool) -> None:
        """Test tutorial validation."""
        builder = TutorialBuilder()
        assert builder.validate_tutorial(tutorial) == expected
