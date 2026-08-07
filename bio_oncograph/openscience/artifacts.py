import json
import logging
import os
from typing import Any, List

logger = logging.getLogger(__name__)

class ArtifactExporter:
    """Exporter for open science artifacts to the filesystem."""
    
    def _ensure_dir(self, directory: str) -> None:
        """Ensure the output directory exists."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

    def export_documentation(self, pages: List[Any], output_dir: str) -> None:
        """Export documentation pages.
        
        Args:
            pages: List of documentation pages to export.
            output_dir: Target directory path.
        """
        self._ensure_dir(output_dir)
        logger.info(f"Exporting {len(pages)} documentation pages to {output_dir}")
        for i, page in enumerate(pages):
            file_path = os.path.join(output_dir, f"page_{i}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(page))
                
    def export_examples(self, examples: List[Any], output_dir: str) -> None:
        """Export code examples.
        
        Args:
            examples: List of examples to export.
            output_dir: Target directory path.
        """
        self._ensure_dir(output_dir)
        logger.info(f"Exporting {len(examples)} examples to {output_dir}")
        for i, example in enumerate(examples):
            file_path = os.path.join(output_dir, f"example_{i}.py")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# Example {i}\n{example}")
                
    def export_reproducibility(self, package: Any, output_dir: str) -> None:
        """Export reproducibility package.
        
        Args:
            package: Reproducibility package data.
            output_dir: Target directory path.
        """
        self._ensure_dir(output_dir)
        logger.info(f"Exporting reproducibility package to {output_dir}")
        file_path = os.path.join(output_dir, "reproducibility_manifest.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(package if isinstance(package, dict) else {"data": str(package)}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to export reproducibility package: {e}")
            
    def export_citations(self, citations: List[Any], output_dir: str) -> None:
        """Export citation data.
        
        Args:
            citations: List of citation objects or strings.
            output_dir: Target directory path.
        """
        self._ensure_dir(output_dir)
        logger.info(f"Exporting {len(citations)} citations to {output_dir}")
        file_path = os.path.join(output_dir, "citations.bib")
        with open(file_path, "w", encoding="utf-8") as f:
            for citation in citations:
                f.write(f"{citation}\n\n")
