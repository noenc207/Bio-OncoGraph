"""
Script to manage releases and publish to PyPI.
"""
import argparse
import logging

from biooncograph.utils.release import ReleaseManager, PyPIPublisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def release_package(version: str, publish: bool) -> None:
    """Bumps version, builds package, and optionally publishes to PyPI.

    Args:
        version: Semantic version string (e.g., '2.0.0').
        publish: If true, publishes to PyPI.
    """
    logger.info(f"Preparing release for version {version}")
    manager = ReleaseManager()
    
    logger.info("Updating version files and changelog")
    manager.bump_version(version)
    
    logger.info("Building source and wheel distributions")
    dist_paths = manager.build_distributions()
    
    if publish:
        logger.info("Publishing to PyPI")
        publisher = PyPIPublisher()
        publisher.publish(dist_paths)
        logger.info("Publish successful.")
    else:
        logger.info("Build complete. Skipping publish step (--publish not provided).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Release Manager")
    parser.add_argument("--version", type=str, required=True, help="New version string")
    parser.add_argument("--publish", action="store_true", help="Publish to PyPI")
    args = parser.parse_args()
    release_package(args.version, args.publish)
