"""
Script to build documentation for BioOncoGraph.
"""
import argparse
import logging
import os

from biooncograph.utils.docs import SphinxBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_documentation(docs_dir: str, output_dir: str, clean: bool) -> None:
    """Builds the sphinx documentation.

    Args:
        docs_dir: Source directory for sphinx docs.
        output_dir: Build output directory.
        clean: Whether to clean the build directory before building.
    """
    if not os.path.exists(docs_dir):
        logger.error(f"Documentation source directory {docs_dir} does not exist.")
        return

    logger.info("Initializing Sphinx Builder")
    builder = SphinxBuilder(source_dir=docs_dir, build_dir=output_dir)
    
    if clean:
        logger.info("Cleaning previous builds")
        builder.clean()
        
    logger.info("Building HTML documentation...")
    builder.build_html()
    logger.info(f"Documentation built successfully in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Documentation")
    parser.add_argument("--docs_dir", type=str, default="docs/source", help="Docs source directory")
    parser.add_argument("--output_dir", type=str, default="docs/build/html", help="Output directory")
    parser.add_argument("--clean", action="store_true", help="Clean before building")
    args = parser.parse_args()
    build_documentation(args.docs_dir, args.output_dir, args.clean)
