import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging

from biooncograph.data.dataset import BioOncoGraphDataset
from biooncograph.models.gnn import BioOncoGraphGNN
from biooncograph.engine.predictor import Predictor

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Starting inference with config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device(cfg.get("predict", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_path = cfg.get("predict", {}).get("output_path", "predictions.json")
    
    logger.info("Initializing dataset for prediction...")
    # For prediction, we might not have labels, assuming 'predict' split or custom data loading
    predict_dataset = BioOncoGraphDataset(cfg.dataset, split="predict")
    predict_loader = predict_dataset.get_dataloader(batch_size=cfg.predict.batch_size, shuffle=False)
    
    logger.info("Initializing model and loading checkpoint...")
    model = BioOncoGraphGNN(cfg.model).to(device)
    checkpoint = torch.load(cfg.predict.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    
    logger.info("Running inference...")
    predictor = Predictor(model, predict_loader, device=device)
    predictions = predictor.predict()
    
    # Save predictions
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=4)
        
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
