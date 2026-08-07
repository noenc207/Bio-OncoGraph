"""
Script to download core datasets for BioOncoGraph.
"""
import argparse
import logging
from typing import List

from biooncograph.data.downloader import DatasetDownloader, BioConductorSource, TCGASource

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_datasets(datasets: List[str], target_dir: str) -> None:
    """Downloads specified biological datasets to a target directory.

    Args:
        datasets: List of dataset identifiers to download.
        target_dir: Destination directory.
    """
    logger.info(f"Initializing downloader for {target_dir}")
    downloader = DatasetDownloader(destination=target_dir)
    
    downloader.register_source("tcga", TCGASource())
    downloader.register_source("bioconductor", BioConductorSource())

    for dataset in datasets:
        logger.info(f"Attempting to download dataset: {dataset}")
        try:
            downloader.download(dataset)
            logger.info(f"Successfully downloaded {dataset}")
        except Exception as e:
            logger.error(f"Failed to download {dataset}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Datasets")
    parser.add_argument("--datasets", nargs="+", required=True, help="Datasets to download")
    parser.add_argument("--target_dir", type=str, default="data/raw", help="Target directory")
    args = parser.parse_args()
    download_datasets(args.datasets, args.target_dir)
