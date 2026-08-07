"""
Example script for training a multimodal model combining imaging and genomics data.
"""
import argparse
import logging
from typing import List

from biooncograph.datasets import MultimodalDataset, DataLoader
from biooncograph.models import MultimodalFusion
from biooncograph.training import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_multimodal(modalities: List[str], data_path: str, epochs: int, output_dir: str) -> None:
    """Trains a multimodal fusion network.

    Args:
        modalities: List of modalities to fuse (e.g., ['genomics', 'imaging']).
        data_path: Base path to the datasets.
        epochs: Number of epochs to train.
        output_dir: Path to save the fused model.
    """
    logger.info(f"Loading modalities: {modalities} from {data_path}")
    dataset = MultimodalDataset(modalities=modalities, data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    logger.info("Building Multimodal Fusion model")
    model = MultimodalFusion(
        modality_encoders={mod: dataset.get_encoder_config(mod) for mod in modalities},
        fusion_type="attention",
        output_dim=dataset.num_classes
    )

    trainer = Trainer(model=model, learning_rate=0.0005)
    logger.info("Beginning multimodal training")
    trainer.fit(dataloader, epochs=epochs)

    model.save(output_dir)
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Training")
    parser.add_argument("--modalities", nargs="+", default=["genomics", "clinical"], help="Modalities to fuse")
    parser.add_argument("--data_path", type=str, default="data/multimodal", help="Path to data")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs")
    parser.add_argument("--output_dir", type=str, default="outputs/multimodal_model", help="Output directory")
    args = parser.parse_args()
    train_multimodal(args.modalities, args.data_path, args.epochs, args.output_dir)
