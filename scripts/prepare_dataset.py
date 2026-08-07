"""
Script to prepare and preprocess datasets for BioOncoGraph models.
"""
import argparse
import logging

from biooncograph.data.preprocessing import DataPreprocessor, GraphFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def prepare_dataset(input_dir: str, output_dir: str, normalize: bool) -> None:
    """Preprocesses raw data into graph representations.

    Args:
        input_dir: Directory containing raw data.
        output_dir: Directory to save preprocessed data.
        normalize: Whether to apply standard normalization.
    """
    logger.info(f"Starting data preparation from {input_dir}")
    
    preprocessor = DataPreprocessor(input_dir=input_dir)
    if normalize:
        logger.info("Applying standard normalizations to features.")
        preprocessor.add_step("standard_scale")
        
    extractor = GraphFeatureExtractor()
    preprocessor.set_extractor(extractor)
    
    logger.info("Processing data...")
    processed_data = preprocessor.process()
    
    logger.info(f"Saving prepared dataset to {output_dir}")
    processed_data.save(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Raw data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Processed data directory")
    parser.add_argument("--normalize", action="store_true", help="Apply normalization")
    args = parser.parse_args()
    prepare_dataset(args.input_dir, args.output_dir, args.normalize)
