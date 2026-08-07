import argparse
import csv
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from bio_oncograph.application.trainer import (
    Trainer,
    TrainerCallback,
    ICheckpoint,
)
from bio_oncograph.core.base import BaseModel, BaseLoss
from bio_oncograph.core.data_models import BioGraphOutput, SurvivalTarget
from bio_oncograph.core.events import EventBus

logger = logging.getLogger(__name__)

# --- Mock Implementations for Self-Contained Execution ---

class MockBatch:
    def __init__(self, size: int):
        self.survival_target = SurvivalTarget(
            time=torch.rand(size),
            event=torch.randint(0, 2, (size,)),
        )

class TCGABrcaDataset(Dataset):
    def __init__(self, size: int = 100):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> MockBatch:
        return MockBatch(1)

def custom_collate(batch: List[MockBatch]) -> MockBatch:
    size = len(batch)
    res = MockBatch(size)
    res.survival_target.time = torch.cat([b.survival_target.time for b in batch])
    res.survival_target.event = torch.cat([b.survival_target.event for b in batch])
    return res

class TCGABrcaModel(BaseModel, nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, data: Any) -> BioGraphOutput:
        return BioGraphOutput(
            node_embeddings=torch.zeros(1),
            graph_embedding=torch.zeros(1),
            hazard_ratio=torch.ones(len(data.survival_target.time), requires_grad=True),
        )

class TCGABrcaLoss(BaseLoss, nn.Module):
    def __init__(self):
        super().__init__()

    class LossOutput:
        def __init__(self, loss):
            self.total_loss = loss

    def forward(self, output: BioGraphOutput, target: SurvivalTarget) -> Any:
        loss = output.hazard_ratio.sum() + target.time.sum() * 0.0
        return self.LossOutput(loss)

# --- Checkpoint Implementation ---

class LocalCheckpoint(ICheckpoint):
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: Dict[str, Any], path: str) -> None:
        file_path = self.checkpoint_dir / f"{path}.pt"
        torch.save(state, file_path)
        
        latest_path = self.checkpoint_dir / "latest.txt"
        with open(latest_path, "w") as f:
            f.write(path)

    def load(self, path: str) -> Dict[str, Any]:
        file_path = self.checkpoint_dir / f"{path}.pt"
        return torch.load(file_path, map_location="cpu", weights_only=False)

    def latest(self) -> Optional[str]:
        latest_path = self.checkpoint_dir / "latest.txt"
        if latest_path.exists():
            with open(latest_path, "r") as f:
                return f.read().strip()
        return None

# --- Callbacks ---

class HistoryCallback(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.history: List[Dict[str, Any]] = []

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: Dict[str, float]) -> None:
        row = {"epoch": epoch}
        row.update(metrics)
        self.history.append(row)

        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        if self.history:
            keys = list(self.history[0].keys())
            with open(self.output_dir / "history.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)

class RunSummaryCallback(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def on_train_end(self, trainer: Trainer) -> None:
        summary_path = self.output_dir / "run_summary.md"
        with open(summary_path, "w") as f:
            f.write("# Training Run Summary\n\n")
            f.write(f"- **Final Epoch**: {trainer.current_epoch}\n")
            f.write(f"- **Global Steps**: {trainer.global_step}\n")
            f.write(f"- **End Time**: {datetime.now().isoformat()}\n")

class CheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_mgr: ICheckpoint):
        self.checkpoint_mgr = checkpoint_mgr
        self.best_loss = float("inf")

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: Dict[str, float]) -> None:
        trainer.save_state(f"epoch_{epoch}")
        
        val_loss = metrics.get("val_loss", float("inf"))
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            trainer.save_state("best")

# --- Main Script ---

def main():
    parser = argparse.ArgumentParser(description="TCGA-BRCA Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./outputs/tcga_brca", help="Output directory")
    parser.add_argument("--use-amp", action="store_true", help="Enable Mixed Precision (AMP)")
    parser.add_argument("--use-ddp", action="store_true", help="Enable Distributed Data Parallel (DDP)")
    parser.add_argument("--resume", type=str, choices=["latest", "best"], help="Resume from checkpoint", default=None)
    parser.add_argument("--resume-epoch", type=int, help="Resume from specific epoch", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "training.log")
        ]
    )

    is_ddp = args.use_ddp
    rank = 0
    if is_ddp:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    model = TCGABrcaModel()
    loss_fn = TCGABrcaLoss()
    
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = TCGABrcaDataset(size=100)
    val_dataset = TCGABrcaDataset(size=20)

    train_sampler = DistributedSampler(train_dataset) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        collate_fn=custom_collate
    )

    checkpoint_mgr = LocalCheckpoint(str(out_dir / "checkpoints"))
    event_bus = EventBus()

    callbacks = [
        HistoryCallback(out_dir),
        RunSummaryCallback(out_dir),
        CheckpointCallback(checkpoint_mgr),
    ]

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        event_bus=event_bus,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint=checkpoint_mgr,
        callbacks=callbacks,
        max_epochs=args.epochs,
        use_amp=args.use_amp,
        sync_ddp=is_ddp,
    )

    # Checkpoint Recovery
    if args.resume_epoch is not None:
        logger.info(f"Resuming from epoch {args.resume_epoch}")
        trainer.load_state(f"epoch_{args.resume_epoch}")
    elif args.resume == "latest":
        latest = checkpoint_mgr.latest()
        if latest:
            logger.info(f"Resuming from latest checkpoint: {latest}")
            trainer.load_state(latest)
        else:
            logger.warning("No latest checkpoint found. Starting from scratch.")
    elif args.resume == "best":
        try:
            logger.info("Resuming from best checkpoint")
            trainer.load_state("best")
        except Exception:
            logger.warning("No best checkpoint found. Starting from scratch.")

    try:
        trainer.fit()
    except Exception as e:
        logger.error("Training crashed!", exc_info=True)
        if rank == 0:
            trainer.save_state("emergency")
            logger.info(f"Emergency checkpoint saved.")
            
            with open(out_dir / "traceback.log", "w") as f:
                traceback.print_exc(file=f)
                
            crash_report = {
                "error": str(e),
                "type": type(e).__name__,
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "timestamp": datetime.now().isoformat()
            }
            with open(out_dir / "crash_report.json", "w") as f:
                json.dump(crash_report, f, indent=2)
        
        if is_ddp:
            dist.destroy_process_group()
        sys.exit(1)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
