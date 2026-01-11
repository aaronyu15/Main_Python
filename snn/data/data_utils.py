"""
Data utilities for loading and preprocessing
"""

import numpy as np
import torch
from typing import Tuple, Optional
import random


def load_events(event_path: str) -> np.ndarray:
    """
    Load event file
    
    Returns:
        Events array [N, 4] with columns [x, y, t, p]
    """
    return np.load(event_path)


def load_flow(flow_path: str) -> np.ndarray:
    """
    Load optical flow file
    
    Returns:
        Flow array [H, W, 2] with channels [u, v]
    """
    return np.load(flow_path)


def augment_data(
    input_tensor: torch.Tensor,
    flow: torch.Tensor,
    valid_mask: torch.Tensor,
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
    rotation: bool = False,
    color_aug: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply data augmentation
    
    Args:
        input_tensor: Input tensor [C, H, W]
        flow: Flow tensor [2, H, W]
        valid_mask: Valid mask [1, H, W]
        horizontal_flip: Enable horizontal flipping
        vertical_flip: Enable vertical flipping
        rotation: Enable small rotations
        color_aug: Enable color augmentation (for RGB inputs)
    
    Returns:
        Augmented (input, flow, valid_mask)
    """
    
    # Horizontal flip
    if horizontal_flip and random.random() > 0.5:
        input_tensor = torch.flip(input_tensor, [2])  # Flip width
        flow = torch.flip(flow, [2])
        flow[0] = -flow[0]  # Flip horizontal flow component
        valid_mask = torch.flip(valid_mask, [2])
    
    # Vertical flip
    if vertical_flip and random.random() > 0.5:
        input_tensor = torch.flip(input_tensor, [1])  # Flip height
        flow = torch.flip(flow, [1])
        flow[1] = -flow[1]  # Flip vertical flow component
        valid_mask = torch.flip(valid_mask, [1])
    
    # Color augmentation (for RGB inputs)
    if color_aug and input_tensor.shape[0] == 3:
        # Random brightness
        brightness_factor = random.uniform(0.8, 1.2)
        input_tensor = input_tensor * brightness_factor
        
        # Random contrast
        contrast_factor = random.uniform(0.8, 1.2)
        mean = input_tensor.mean()
        input_tensor = (input_tensor - mean) * contrast_factor + mean
        
        # Clamp to valid range
        input_tensor = torch.clamp(input_tensor, 0, 1)
    
    return input_tensor, flow, valid_mask


def random_crop(
    input_tensor: torch.Tensor,
    flow: torch.Tensor,
    valid_mask: torch.Tensor,
    crop_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply random crop
    
    Args:
        input_tensor: Input tensor [C, H, W]
        flow: Flow tensor [2, H, W]
        valid_mask: Valid mask [1, H, W]
        crop_size: (height, width) to crop to
    
    Returns:
        Cropped (input, flow, valid_mask)
    """
    _, h, w = input_tensor.shape
    crop_h, crop_w = crop_size
    
    # Random crop position
    if h > crop_h:
        start_h = random.randint(0, h - crop_h)
    else:
        start_h = 0
    
    if w > crop_w:
        start_w = random.randint(0, w - crop_w)
    else:
        start_w = 0
    
    # Crop
    input_tensor = input_tensor[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
    flow = flow[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
    valid_mask = valid_mask[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    return input_tensor, flow, valid_mask


def normalize_events(events: torch.Tensor) -> torch.Tensor:
    """
    Normalize event voxel grid
    
    Args:
        events: Event tensor [C, H, W]
    
    Returns:
        Normalized events
    """
    # Normalize to zero mean, unit variance per channel
    for c in range(events.shape[0]):
        mean = events[c].mean()
        std = events[c].std()
        if std > 0:
            events[c] = (events[c] - mean) / std
    
    return events


def create_event_mask(events: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """
    Create mask for regions with sufficient event activity
    
    Args:
        events: Event tensor [C, H, W]
        threshold: Minimum activity threshold
    
    Returns:
        Mask tensor [1, H, W]
    """
    # Sum absolute values across channels
    activity = torch.abs(events).sum(dim=0, keepdim=True)
    
    # Create mask
    mask = (activity > threshold).float()
    
    return mask


class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, input_tensor, flow, valid_mask):
        for transform in self.transforms:
            input_tensor, flow, valid_mask = transform(input_tensor, flow, valid_mask)
        return input_tensor, flow, valid_mask


class RandomHorizontalFlip:
    """Random horizontal flip transform"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, input_tensor, flow, valid_mask):
        if random.random() < self.p:
            input_tensor = torch.flip(input_tensor, [2])
            flow = torch.flip(flow, [2])
            flow[0] = -flow[0]
            valid_mask = torch.flip(valid_mask, [2])
        return input_tensor, flow, valid_mask


class RandomCrop:
    """Random crop transform"""
    def __init__(self, crop_size):
        self.crop_size = crop_size
    
    def __call__(self, input_tensor, flow, valid_mask):
        return random_crop(input_tensor, flow, valid_mask, self.crop_size)


class Normalize:
    """Normalize transform"""
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    
    def __call__(self, input_tensor, flow, valid_mask):
        if self.mean is not None and self.std is not None:
            for c in range(input_tensor.shape[0]):
                input_tensor[c] = (input_tensor[c] - self.mean[c]) / self.std[c]
        else:
            # Auto normalize
            input_tensor = normalize_events(input_tensor)
        
        return input_tensor, flow, valid_mask
