"""Tests for ReleaseValidator."""

import pytest
import re
from typing import List, Dict, Any


class ReleaseValidator:
    """Mock ReleaseValidator for testing."""

    def __init__(self, version_pattern: str = r"^\d+\.\d+\.\d+$") -> None:
        self.version_pattern = re.compile(version_pattern)

    def verify_version_string(self, version: str) -> bool:
        """Verify version string format."""
        return bool(self.version_pattern.match(version))

    def verify_package_structure(self, files: List[str]) -> bool:
        """Verify essential package files are present."""
        required = {"setup.py", "README.md", "LICENSE"}
        return required.issubset(set(files))

    def check_dependencies(self, dependencies: Dict[str, str]) -> List[str]:
        """Check dependencies and return list of invalid ones."""
        invalid = []
        for dep, version in dependencies.items():
            if not self.verify_version_string(version):
                invalid.append(dep)
        return invalid


@pytest.fixture
def validator() -> ReleaseValidator:
    return ReleaseValidator()


def test_validator_init(validator: ReleaseValidator) -> None:
    assert validator.version_pattern is not None


def test_verify_version_string_valid(validator: ReleaseValidator) -> None:
    assert validator.verify_version_string("1.0.0") is True


def test_verify_version_string_invalid_letters(validator: ReleaseValidator) -> None:
    assert validator.verify_version_string("1.0.a") is False


def test_verify_version_string_missing_patch(validator: ReleaseValidator) -> None:
    assert validator.verify_version_string("1.0") is False


def test_verify_version_string_extra_parts(validator: ReleaseValidator) -> None:
    assert validator.verify_version_string("1.0.0.0") is False


def test_verify_version_string_beta(validator: ReleaseValidator) -> None:
    assert validator.verify_version_string("1.0.0-beta") is False


def test_verify_package_structure_valid(validator: ReleaseValidator) -> None:
    files = ["setup.py", "README.md", "LICENSE", "src/"]
    assert validator.verify_package_structure(files) is True


def test_verify_package_structure_missing_setup(validator: ReleaseValidator) -> None:
    files = ["README.md", "LICENSE", "src/"]
    assert validator.verify_package_structure(files) is False


def test_verify_package_structure_missing_readme(validator: ReleaseValidator) -> None:
    files = ["setup.py", "LICENSE", "src/"]
    assert validator.verify_package_structure(files) is False


def test_verify_package_structure_missing_license(validator: ReleaseValidator) -> None:
    files = ["setup.py", "README.md", "src/"]
    assert validator.verify_package_structure(files) is False


def test_verify_package_structure_empty(validator: ReleaseValidator) -> None:
    assert validator.verify_package_structure([]) is False


def test_verify_package_structure_exact(validator: ReleaseValidator) -> None:
    files = ["setup.py", "README.md", "LICENSE"]
    assert validator.verify_package_structure(files) is True


def test_check_dependencies_all_valid(validator: ReleaseValidator) -> None:
    deps = {"numpy": "1.21.0", "pandas": "1.3.0"}
    assert validator.check_dependencies(deps) == []


def test_check_dependencies_one_invalid(validator: ReleaseValidator) -> None:
    deps = {"numpy": "1.21.0", "pandas": "1.3"}
    assert validator.check_dependencies(deps) == ["pandas"]


def test_check_dependencies_all_invalid(validator: ReleaseValidator) -> None:
    deps = {"numpy": "1.21", "pandas": "1.3"}
    assert validator.check_dependencies(deps) == ["numpy", "pandas"]


def test_check_dependencies_empty(validator: ReleaseValidator) -> None:
    assert validator.check_dependencies({}) == []


def test_check_dependencies_invalid_format(validator: ReleaseValidator) -> None:
    deps = {"pkg": "latest"}
    assert validator.check_dependencies(deps) == ["pkg"]


def test_validator_custom_pattern() -> None:
    val = ReleaseValidator(version_pattern=r"^\d+\.\d+$")
    assert val.verify_version_string("1.0") is True
    assert val.verify_version_string("1.0.0") is False


def test_verify_version_string_empty(validator: ReleaseValidator) -> None:
    assert validator.verify_version_string("") is False


def test_verify_version_string_whitespace(validator: ReleaseValidator) -> None:
    assert validator.verify_version_string(" 1.0.0 ") is False


def test_check_dependencies_with_valid_and_invalid(validator: ReleaseValidator) -> None:
    deps = {"valid1": "1.0.0", "invalid1": "1.0", "valid2": "2.0.0"}
    assert validator.check_dependencies(deps) == ["invalid1"]
