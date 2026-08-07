"""Dataset metadata definitions."""

import dataclasses
from typing import Any, Dict


@dataclasses.dataclass
class DatasetMetadata:
    """Metadata for a dataset.

    Attributes:
        name: The name of the dataset.
        version: The version of the dataset.
        modality: The modality of the dataset (e.g., 'genomics', 'imaging').
        schema: A dictionary describing the schema of the dataset.
        citation: Citation information for the dataset.
        license: The license under which the dataset is provided.
        source: The source URL or location of the dataset.
        checksum: The checksum (e.g., MD5 or SHA256) of the dataset files.
    """
    name: str
    version: str
    modality: str
    schema: Dict[str, Any]
    citation: str
    license: str
    source: str
    checksum: str
