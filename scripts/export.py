import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
import json

from biooncograph.models.gnn import BioOncoGraphGNN

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Starting export with config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device("cpu") # Export usually done on CPU
    checkpoint_path = cfg.export.checkpoint_path
    export_dir = cfg.get("export", {}).get("export_dir", "exported_model")
    export_format = cfg.get("export", {}).get("format", "onnx")
    
    os.makedirs(export_dir, exist_ok=True)
    
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    model = BioOncoGraphGNN(cfg.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    
    # Save metadata
    metadata = {
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        "input_features": cfg.get("dataset", {}).get("num_features", None),
        "num_classes": cfg.get("dataset", {}).get("num_classes", None),
        "version": "1.0"
    }
    with open(os.path.join(export_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info("Exporting model weights...")
    if export_format == "torchscript":
        # Example dummy input needed for tracing, might need adjustment based on exact model input
        dummy_input = (torch.randn(1, metadata["input_features"]), torch.randint(0, 1, (2, 10))) 
        try:
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(os.path.join(export_dir, "model.pt"))
            logger.info("Successfully exported to TorchScript.")
        except Exception as e:
            logger.warning(f"TorchScript export failed, fallback to state_dict: {e}")
            torch.save(model.state_dict(), os.path.join(export_dir, "model_weights.pth"))
            
    elif export_format == "onnx":
        dummy_input = (torch.randn(1, metadata["input_features"]), torch.randint(0, 1, (2, 10)))
        try:
            torch.onnx.export(
                model, 
                dummy_input, 
                os.path.join(export_dir, "model.onnx"),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['x', 'edge_index'],
                output_names=['output'],
                dynamic_axes={'x': {0: 'batch_size'}, 'edge_index': {1: 'num_edges'}, 'output': {0: 'batch_size'}}
            )
            logger.info("Successfully exported to ONNX.")
        except Exception as e:
            logger.warning(f"ONNX export failed, fallback to state_dict: {e}")
            torch.save(model.state_dict(), os.path.join(export_dir, "model_weights.pth"))
    else:
        # Default save state dict
        torch.save(model.state_dict(), os.path.join(export_dir, "model_weights.pth"))
        logger.info("Saved raw state dict.")
        
    logger.info(f"Export complete. Files saved to {export_dir}")

if __name__ == "__main__":
    main()
