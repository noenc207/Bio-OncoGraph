"""Validation utilities for dataset connectors."""

import hashlib
import os
from typing import Dict, List, Set


class DatasetValidator:
    """Validator to check dataset integrity."""

    @staticmethod
    def check_missing_files(expected_files: List[str], base_dir: str) -> List[str]:
        """Checks for missing files.

        Args:
            expected_files: List of expected file paths relative to base_dir.
            base_dir: The base directory to check within.

        Returns:
            A list of missing file paths.
        """
        missing = []
        for file_path in expected_files:
            full_path = os.path.join(base_dir, file_path)
            if not os.path.exists(full_path):
                missing.append(file_path)
        return missing

    @staticmethod
    def check_corrupted_files(file_checksums: Dict[str, str], base_dir: str) -> List[str]:
        """Checks for corrupted files based on checksums.

        Args:
            file_checksums: Dictionary mapping file paths to expected checksums.
            base_dir: The base directory to check within.

        Returns:
            A list of corrupted file paths.
        """
        corrupted = []
        for file_path, expected_checksum in file_checksums.items():
            full_path = os.path.join(base_dir, file_path)
            if not os.path.exists(full_path):
                continue
            
            hasher = hashlib.md5()
            with open(full_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            if hasher.hexdigest() != expected_checksum:
                corrupted.append(file_path)
        return corrupted

    @staticmethod
    def check_inconsistent_annotations(annotations: Dict[str, str], allowed_values: Set[str]) -> List[str]:
        """Checks for inconsistent annotations.

        Args:
            annotations: Dictionary mapping identifiers to annotation values.
            allowed_values: Set of allowed annotation values.

        Returns:
            A list of identifiers with inconsistent annotations.
        """
        return [k for k, v in annotations.items() if v not in allowed_values]

    @staticmethod
    def check_duplicated_identifiers(identifiers: List[str]) -> List[str]:
        """Checks for duplicated identifiers.

        Args:
            identifiers: List of identifiers to check.

        Returns:
            A list of duplicated identifiers.
        """
        seen: Set[str] = set()
        duplicates = set()
        for identifier in identifiers:
            if identifier in seen:
                duplicates.add(identifier)
            seen.add(identifier)
        return list(duplicates)

    @staticmethod
    def check_invalid_splits(splits: Dict[str, List[str]], all_identifiers: List[str]) -> bool:
        """Checks if dataset splits are valid (mutually exclusive and exhaustive).

        Args:
            splits: Dictionary mapping split names to lists of identifiers.
            all_identifiers: List of all valid identifiers.

        Returns:
            True if splits are valid, False otherwise.
        """
        all_split_ids: Set[str] = set()
        for split_ids in splits.values():
            split_set = set(split_ids)
            if not all_split_ids.isdisjoint(split_set):
                return False
            all_split_ids.update(split_set)
        
        return all_split_ids == set(all_identifiers)
