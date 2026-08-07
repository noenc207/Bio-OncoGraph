"""
Example script for training a survival analysis model using BioOncoGraph.
"""
import argparse
import logging

from biooncograph.datasets import SurvivalDataset, DataLoader
from biooncograph.models import CoxPHGraphModel
from biooncograph.training import SurvivalTrainer
from biooncograph.metrics import ConcordanceIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_survival_model(data_path: str, epochs: int, output_dir: str) -> None:
    """Trains a Cox Proportional Hazards model on graph data.

    Args:
        data_path: Path to survival dataset.
        epochs: Number of training epochs.
        output_dir: Save location for the model.
    """
    logger.info(f"Loading survival data from {data_path}")
    dataset = SurvivalDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    logger.info("Initializing CoxPH Graph Model")
    model = CoxPHGraphModel(hidden_dim=64, num_layers=2)

    logger.info("Setting up Survival Trainer")
    trainer = SurvivalTrainer(model=model, learning_rate=0.001)

    logger.info("Training...")
    trainer.fit(dataloader, epochs=epochs)

    logger.info("Computing Concordance Index on training set")
    predictions, true_times, events = trainer.predict(dataloader)
    c_index = ConcordanceIndex()(predictions, true_times, events)
    logger.info(f"Final C-Index: {c_index:.4f}")

    model.save(output_dir)
    logger.info(f"Survival model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Survival Model Training")
    parser.add_argument("--data_path", type=str, default="data/survival", help="Path to data")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs")
    parser.add_argument("--output_dir", type=str, default="outputs/survival_model", help="Output directory")
    args = parser.parse_args()
    train_survival_model(args.data_path, args.epochs, args.output_dir)
