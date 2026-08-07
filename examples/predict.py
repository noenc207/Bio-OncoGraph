"""
Example script for running inference/predictions with a trained BioOncoGraph model.
"""
import argparse
import logging
import json

from biooncograph.inference import Predictor
from biooncograph.datasets import InferenceDataset
from biooncograph.models import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_predictions(model_path: str, data_path: str, output_path: str) -> None:
    """Generates predictions for a set of new samples.

    Args:
        model_path: Path to the saved model.
        data_path: Path to the unseen samples.
        output_path: Path to save prediction results.
    """
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    logger.info(f"Loading inference dataset from {data_path}")
    dataset = InferenceDataset(data_path=data_path)
    
    predictor = Predictor(model=model)
    logger.info("Generating predictions...")
    predictions = predictor.predict(dataset)
    
    logger.info(f"Saving predictions to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions.to_dict(), f, indent=4)
        
    logger.info("Prediction generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Predictions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to unseen data")
    parser.add_argument("--output_path", type=str, default="predictions.json", help="Output file")
    args = parser.parse_args()
    run_predictions(args.model_path, args.data_path, args.output_path)
