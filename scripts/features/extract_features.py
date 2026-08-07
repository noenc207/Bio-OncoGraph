import os
import argparse
import logging
import h5py
import lmdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import autocast
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import timm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self):
        self.models = {}
        
    def register(self, name):
        def decorator(cls):
            self.models[name] = cls
            return cls
        return decorator
        
    def get_model(self, name, **kwargs):
        if name not in self.models:
            raise ValueError(f"Model {name} not found in registry.")
        return self.models[name](**kwargs)

registry = ModelRegistry()

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def forward(self, x):
        raise NotImplementedError

@registry.register("UNI")
class UNIModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
    def forward(self, x):
        return self.model(x)

@registry.register("UNI2")
class UNI2Model(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model("vit_huge_patch14_224", pretrained=False, num_classes=0)
    def forward(self, x):
        return self.model(x)

@registry.register("CONCH")
class CONCHModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    def forward(self, x):
        return self.model(x)

@registry.register("Virchow")
class VirchowModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model("vit_huge_patch14_224", pretrained=False, num_classes=0)
    def forward(self, x):
        return self.model(x)

@registry.register("Virchow2")
class Virchow2Model(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model("vit_huge_patch14_224", pretrained=False, num_classes=0)
    def forward(self, x):
        return self.model(x)

@registry.register("Phikon")
class PhikonModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    def forward(self, x):
        return self.model(x)

@registry.register("Prov-GigaPath")
class ProvGigaPathModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
    def forward(self, x):
        return self.model(x)

@registry.register("PLIP")
class PLIPModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    def forward(self, x):
        return self.model(x)

@registry.register("CHIEF")
class CHIEFModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
    def forward(self, x):
        return self.model(x)

@registry.register("TITAN")
class TITANModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
    def forward(self, x):
        return self.model(x)

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            tensor = self.transform(image)
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            tensor = torch.zeros(3, 224, 224)
        return tensor, str(path)

def save_features(output_dir, features, paths, fmt="pt", rank=0):
    os.makedirs(output_dir, exist_ok=True)
    if fmt == "pt":
        for feat, path in zip(features, paths):
            name = Path(path).stem
            torch.save(feat, os.path.join(output_dir, f"{name}.pt"))
    elif fmt == "h5":
        h5_path = os.path.join(output_dir, f"features_rank{rank}.h5")
        with h5py.File(h5_path, "w") as f:
            for feat, path in zip(features, paths):
                name = Path(path).stem
                f.create_dataset(name, data=feat.numpy())
    elif fmt == "lmdb":
        lmdb_path = os.path.join(output_dir, f"features_rank{rank}.lmdb")
        env = lmdb.open(lmdb_path, map_size=1099511627776)
        with env.begin(write=True) as txn:
            for feat, path in zip(features, paths):
                name = Path(path).stem.encode('utf-8')
                txn.put(name, feat.numpy().tobytes())
        env.close()
    else:
        raise ValueError(f"Unknown format {fmt}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(registry.models.keys()))
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--format", type=str, default="pt", choices=["pt", "h5", "lmdb"])
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    args = parser.parse_args()

    is_distributed = args.local_rank != -1
    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        
    device = torch.device(f'cuda:{args.local_rank}' if is_distributed else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    model = registry.get_model(args.model)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        
    image_paths = list(Path(args.data_dir).glob('**/*.*'))
    image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']]
    
    dataset = ImageDataset(image_paths, model.module.transform if is_distributed else model.transform)
    
    sampler = DistributedSampler(dataset) if is_distributed else None
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    all_features = []
    all_paths = []
    
    with torch.no_grad():
        for batch_idx, (images, paths) in enumerate(dataloader):
            images = images.to(device)
            with autocast():
                features = model(images)
                
            all_features.append(features.cpu())
            all_paths.extend(paths)
            
            if args.local_rank in [-1, 0] and batch_idx % 10 == 0:
                logger.info(f"Processed batch {batch_idx}/{len(dataloader)}")
                
    if len(all_features) > 0:
        all_features = torch.cat(all_features, dim=0)
    else:
        all_features = torch.empty((0,))
        
    save_features(args.output_dir, all_features, all_paths, fmt=args.format, rank=max(0, args.local_rank))
    logger.info(f"Rank {max(0, args.local_rank)}: Saved features to {args.output_dir} in {args.format} format.")

if __name__ == "__main__":
    main()
