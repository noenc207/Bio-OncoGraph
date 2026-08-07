import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from biooncograph.data.dataset import BioOncoGraphDataset
from biooncograph.models.gnn import BioOncoGraphGNN
from biooncograph.engine.trainer import Trainer

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Starting training with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Setup DDP and AMP
    use_ddp = cfg.get("training", {}).get("use_ddp", False)
    use_amp = cfg.get("training", {}).get("use_amp", False)
    device_str = cfg.get("training", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    if use_ddp:
        torch.distributed.init_process_group(backend="nccl" if "cuda" in device_str else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if "cuda" in device_str:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    logger.info("Initializing datasets...")
    train_dataset = BioOncoGraphDataset(cfg.dataset, split="train")
    val_dataset = BioOncoGraphDataset(cfg.dataset, split="val")
    
    train_loader = train_dataset.get_dataloader(batch_size=cfg.training.batch_size, shuffle=True, use_ddp=use_ddp)
    val_loader = val_dataset.get_dataloader(batch_size=cfg.training.batch_size, shuffle=False, use_ddp=use_ddp)
    
    logger.info("Initializing model...")
    model = BioOncoGraphGNN(cfg.model).to(device)
    
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if "cuda" in device_str else None)
        
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=use_amp,
        epochs=cfg.training.epochs,
        checkpoint_dir=cfg.training.checkpoint_dir
    )
    
    logger.info("Starting training loop...")
    trainer.train()
    logger.info("Training complete.")
    
    if use_ddp:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
