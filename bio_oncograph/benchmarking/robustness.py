import torch
import torchvision.transforms as T
import torch.nn as nn
from PIL import ImageFilter, Image
import io

class GaussianNoiseTransform:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class BlurTransform:
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            import torchvision.transforms.functional as F
            return F.gaussian_blur(img, kernel_size=self.radius * 2 + 1)
        return img.filter(ImageFilter.GaussianBlur(self.radius))

class JPEGCompressionTransform:
    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            import torchvision.transforms.functional as F
            img = F.to_pil_image(img)
            
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=self.quality)
        buffer.seek(0)
        img = Image.open(buffer)
        
        return T.ToTensor()(img)

class RobustDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms=None, missing_modality_prob=0.0):
        self.dataset = dataset
        self.transforms = transforms
        self.missing_modality_prob = missing_modality_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        if isinstance(data, tuple) and len(data) == 2:
            inputs, labels = data
            
            if isinstance(inputs, dict):
                new_inputs = {}
                for k, v in inputs.items():
                    if torch.rand(1).item() < self.missing_modality_prob:
                        new_inputs[k] = torch.zeros_like(v)
                    else:
                        if self.transforms:
                            new_inputs[k] = self.transforms(v)
                        else:
                            new_inputs[k] = v
                inputs = new_inputs
            else:
                if self.transforms:
                    inputs = self.transforms(inputs)
            
            return inputs, labels
            
        return data
