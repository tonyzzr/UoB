"""Custom torchvision transforms."""

import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import InterpolationMode
from PIL import Image
import math

class PadToSquareAndAlign(torch.nn.Module):
    """Pads a PIL Image or Tensor to a target square size, aligning to top-center.

    The input image is placed centered horizontally and aligned to the top vertically
    within the target square dimensions. Padding is added to the bottom and sides.

    Args:
        size (int): The target height and width of the output square image.
        fill (int or float): Pixel fill value for the padding. Defaults to 0 (black).
        padding_mode (str): Type of padding. Should be: 'constant', 'edge',
            'reflect' or 'symmetric'. Defaults to 'constant'.
    """
    def __init__(self, size: int, fill: int | float = 0, padding_mode: str = 'constant'):
        super().__init__()
        if not isinstance(size, int):
            raise TypeError("Size must be an integer.")
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img: torch.Tensor | Image.Image) -> torch.Tensor:
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            Tensor: Padded image.
        """
        # Convert PIL Image to Tensor using recommended v2 functions
        if isinstance(img, Image.Image):
             # Convert PIL to tensor (C, H, W) with uint8 dtype first
             img_tensor = F.to_image(img)
             # Convert to float32 and scale to [0, 1]
             img_tensor = F.to_dtype(img_tensor, dtype=torch.float32, scale=True)
        elif isinstance(img, torch.Tensor):
             img_tensor = img
        else:
             raise TypeError("Input must be a PIL Image or a torch.Tensor")

        _ , h, w = F.get_dimensions(img_tensor)

        if h > self.size or w > self.size:
            # If the image is larger than the target size in any dimension,
            # resize it while maintaining aspect ratio so the longest side fits.
            # This might not be strictly necessary if inputs are always smaller,
            # but adds robustness.
            scale_factor = self.size / max(h, w)
            new_h = math.floor(h * scale_factor)
            new_w = math.floor(w * scale_factor)
            img_tensor = F.resize(img_tensor, [new_h, new_w], interpolation=InterpolationMode.BICUBIC, antialias=True)
            _ , h, w = F.get_dimensions(img_tensor)

        if h == self.size and w == self.size:
             return img_tensor # Already target size

        pad_h = self.size - h
        pad_w = self.size - w

        # Calculate padding: [left, top, right, bottom]
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = 0 # Align to top
        pad_bottom = pad_h

        padding = [pad_left, pad_top, pad_right, pad_bottom]

        return F.pad(img_tensor, padding, fill=self.fill, padding_mode=self.padding_mode)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, fill={self.fill}, padding_mode='{self.padding_mode}')" 