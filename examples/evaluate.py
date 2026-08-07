"""
Example script for evaluating a trained model on a test set.
"""
import argparse
import logging

from biooncograph.evaluation import ModelEvaluator
from biooncograph.datasets import TestDataset, DataLoader
from biooncograph.models import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model(model_path: str, data_path: str) -> None:
    """Evaluates a saved model on a test dataset.

    Args:
        model_path: Path to the saved model weights.
        data_path: Path to the test dataset.
    """
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)

    logger.info(f"Loading test data from {data_path}")
    dataset = TestDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    logger.info("Running evaluation")
    evaluator = ModelEvaluator(model=model)
    results = evaluator.evaluate(dataloader)
    
    logger.info("Evaluation Results:")
    for metric_name, value in results.items():
        logger.info(f"  {metric_name}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data")
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_path)
