"""General utility functions for dataset management."""

import os
from typing import Any, Generator, List


def list_files_with_extension(directory: str, extension: str) -> List[str]:
    """Lists all files in a directory with a specific extension.

    Args:
        directory: The directory to search.
        extension: The file extension to filter by (e.g., '.csv').

    Returns:
        A list of file paths.
    """
    matched_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                matched_files.append(os.path.join(root, file))
    return matched_files


def batched(iterable: List[Any], n: int) -> Generator[List[Any], None, None]:
    """Yields successive n-sized chunks from an iterable.

    Args:
        iterable: The iterable to chunk.
        n: The size of each chunk.

    Yields:
        Chunks of size n.
    """
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def ensure_directory_exists(directory_path: str) -> None:
    """Ensures that a directory exists, creating it if necessary.

    Args:
        directory_path: The path of the directory.
    """
    os.makedirs(directory_path, exist_ok=True)
