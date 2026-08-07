import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    SequentialLR
)

class SchedulerRegistry:
    @staticmethod
    def get_scheduler(name: str, optimizer, **kwargs):
        name_lower = name.lower()
        if name_lower == "cosineannealinglr":
            return CosineAnnealingLR(optimizer, **kwargs)
        elif name_lower == "reducelronplateau":
            return ReduceLROnPlateau(optimizer, **kwargs)
        elif name_lower == "onecyclelr":
            return OneCycleLR(optimizer, **kwargs)
        elif name_lower == "sequentiallr":
            return SequentialLR(optimizer, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {name}")
