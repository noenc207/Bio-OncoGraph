import os
import numpy as np
from PIL import Image
from typing import Tuple, List

class WSIReader:
    def __init__(self, slide_path: str):
        self.slide_path = slide_path
        self.backend = None
        self.slide = None
        self._load_slide()

    def _load_slide(self):
        try:
            import cucim
            self.slide = cucim.CuImage(self.slide_path)
            self.backend = 'cucim'
            return
        except Exception:
            pass

        try:
            import tiffslide
            self.slide = tiffslide.TiffSlide(self.slide_path)
            self.backend = 'tiffslide'
            return
        except Exception:
            pass

        try:
            import openslide
            self.slide = openslide.OpenSlide(self.slide_path)
            self.backend = 'openslide'
            return
        except Exception:
            pass

        raise RuntimeError(f"Failed to load slide {self.slide_path} with cuCIM, TiffSlide, or OpenSlide.")

    @property
    def dimensions(self) -> Tuple[int, int]:
        if self.backend == 'cucim':
            shape = self.slide.metadata.get('cucim', {}).get('shape', self.slide.shape)
            return shape[1::-1] if len(shape) >= 2 else (0, 0)
        return self.slide.dimensions

    @property
    def level_count(self) -> int:
        if self.backend == 'cucim':
            return self.slide.resolutions.get('level_count', len(self.slide.resolutions.get('levels', [1]))) if hasattr(self.slide, 'resolutions') else 1
        return self.slide.level_count

    @property
    def level_dimensions(self) -> Tuple[Tuple[int, int], ...]:
        if self.backend == 'cucim':
            if hasattr(self.slide, 'resolutions') and 'levels' in self.slide.resolutions:
                return tuple((lvl['width'], lvl['height']) for lvl in self.slide.resolutions['levels'])
            return (self.dimensions,)
        return self.slide.level_dimensions

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> Image.Image:
        if self.backend == 'cucim':
            arr = np.asarray(self.slide.read_region(location=location, size=size, level=level))
            return Image.fromarray(arr).convert("RGB")
        return self.slide.read_region(location, level, size).convert("RGB")

    def close(self):
        if hasattr(self.slide, 'close'):
            self.slide.close()
