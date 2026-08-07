"""
Example script for training a graph neural network on molecular data.
"""
import argparse
import logging

from biooncograph.datasets import GraphDataset, DataLoader
from biooncograph.models import GNNClassifier
from biooncograph.training import Trainer
from biooncograph.evaluation import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_graph_model(data_path: str, epochs: int, batch_size: int, output_dir: str) -> None:
    """Trains a GNN model on the provided graph dataset.

    Args:
        data_path: Path to the dataset directory.
        epochs: Number of training epochs.
        batch_size: Batch size for DataLoader.
        output_dir: Directory to save the trained model.
    """
    logger.info(f"Loading graph dataset from {data_path}")
    dataset = GraphDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info("Initializing GNN Classifier model")
    model = GNNClassifier(hidden_dim=128, num_layers=3, num_classes=dataset.num_classes)

    logger.info("Setting up Trainer")
    trainer = Trainer(model=model, learning_rate=0.001)

    logger.info(f"Starting training for {epochs} epochs")
    trainer.fit(dataloader, epochs=epochs)

    logger.info("Evaluating model")
    evaluator = Evaluator(model=model)
    metrics = evaluator.evaluate(dataloader)
    logger.info(f"Training metrics: {metrics}")

    logger.info(f"Saving model to {output_dir}")
    model.save(output_dir)
    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph Model")
    parser.add_argument("--data_path", type=str, default="data/graph_data", help="Path to data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="outputs/model", help="Output directory")
    args = parser.parse_args()
    train_graph_model(args.data_path, args.epochs, args.batch_size, args.output_dir)
