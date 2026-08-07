import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SemanticVersion:
    """Represents a semantic version.

    Attributes:
        major: Major version number for incompatible API changes.
        minor: Minor version number for backward-compatible functionality.
        patch: Patch version number for backward-compatible bug fixes.
    """
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        """Returns the string representation of the version."""
        return f"{self.major}.{self.minor}.{self.patch}"


class VersionManager:
    """Manages semantic version parsing and incrementing."""

    @staticmethod
    def parse_version(version_string: str) -> SemanticVersion:
        """Parses a version string into a SemanticVersion object.

        Args:
            version_string: The version string to parse (e.g., '1.2.3').

        Returns:
            A SemanticVersion object representing the parsed version.

        Raises:
            ValueError: If the version string is invalid.
        """
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_string)
        if not match:
            msg = f"Invalid semantic version string: {version_string}"
            logger.error(msg)
            raise ValueError(msg)
        
        major, minor, patch = map(int, match.groups())
        return SemanticVersion(major=major, minor=minor, patch=patch)

    @staticmethod
    def increment_version(
        version: SemanticVersion, increment_type: str
    ) -> SemanticVersion:
        """Increments a semantic version based on the specified type.

        Args:
            version: The current SemanticVersion.
            increment_type: The type of increment ('major', 'minor', or 'patch').

        Returns:
            A new SemanticVersion object with the incremented value.

        Raises:
            ValueError: If the increment_type is invalid.
        """
        if increment_type == "major":
            new_version = SemanticVersion(major=version.major + 1, minor=0, patch=0)
        elif increment_type == "minor":
            new_version = SemanticVersion(major=version.major, minor=version.minor + 1, patch=0)
        elif increment_type == "patch":
            new_version = SemanticVersion(major=version.major, minor=version.minor, patch=version.patch + 1)
        else:
            msg = f"Invalid increment type: {increment_type}. Must be 'major', 'minor', or 'patch'."
            logger.error(msg)
            raise ValueError(msg)
        
        logger.info(f"Incremented version from {version} to {new_version} ({increment_type}).")
        return new_version
