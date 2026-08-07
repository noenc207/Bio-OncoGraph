"""
Example script for pre-training a foundation model on unlabelled biomedical graphs.
"""
import argparse
import logging

from biooncograph.foundation import FoundationModel, Pretrainer
from biooncograph.datasets import LargeScaleUnlabeledDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_foundation_model(data_path: str, epochs: int, output_dir: str) -> None:
    """Trains a foundation model using self-supervised learning.

    Args:
        data_path: Path to the unlabelled dataset.
        epochs: Number of pretraining epochs.
        output_dir: Directory to save the pre-trained weights.
    """
    logger.info(f"Loading large scale unlabelled data from {data_path}")
    dataset = LargeScaleUnlabeledDataset(data_path=data_path)
    
    logger.info("Initializing Foundation Model backbone")
    model = FoundationModel(embedding_dim=256, num_layers=6, attention_heads=8)
    
    logger.info("Configuring Self-Supervised Pretrainer")
    pretrainer = Pretrainer(
        model=model, 
        masking_ratio=0.15,
        objective="graph_reconstruction"
    )
    
    logger.info(f"Starting pre-training for {epochs} epochs")
    pretrainer.fit(dataset, epochs=epochs)
    
    logger.info(f"Saving pre-trained foundation model to {output_dir}")
    model.save(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foundation Model Pretraining")
    parser.add_argument("--data_path", type=str, required=True, help="Path to unlabelled data")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--output_dir", type=str, default="outputs/foundation", help="Output directory")
    args = parser.parse_args()
    train_foundation_model(args.data_path, args.epochs, args.output_dir)
