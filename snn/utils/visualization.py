"""
Visualization utilities for optical flow
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path
from typing import Optional, Tuple


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color(flow: np.ndarray, max_flow: Optional[float] = None, clip_flow: Optional[float] = None) -> np.ndarray:
    """
    Convert optical flow to color visualization using the Middlebury color scheme
    
    Args:
        flow: Flow array [H, W, 2] or [2, H, W]
        max_flow: Maximum flow magnitude for normalization (deprecated, use clip_flow)
        clip_flow: Clip maximum of flow values
    
    Returns:
        Color image [H, W, 3] in range [0, 255]
    """
    # Convert to [H, W, 2] format
    if flow.shape[0] == 2:
        flow = np.transpose(flow, (1, 2, 0))
    
    assert flow.ndim == 3, 'input flow must have three dimensions'
    assert flow.shape[2] == 2, 'input flow must have shape [H,W,2]'
    
    if clip_flow is not None:
        flow = np.clip(flow, -clip_flow, clip_flow)
    
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    
    return flow_uv_to_colors(u, v, convert_to_bgr=False)


def visualize_flow(flow: torch.Tensor, max_flow: Optional[float] = None) -> np.ndarray:
    """
    Visualize flow tensor
    
    Args:
        flow: Flow tensor [2, H, W] or [B, 2, H, W]
        max_flow: Maximum flow for normalization
    
    Returns:
        Color visualization [H, W, 3] or [B, H, W, 3]
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    
    if flow.ndim == 4:
        # Batch of flows
        batch_size = flow.shape[0]
        visualizations = []
        for i in range(batch_size):
            vis = flow_to_color(flow[i], max_flow)
            visualizations.append(vis)
        return np.stack(visualizations)
    else:
        # Single flow
        return flow_to_color(flow, max_flow)


def save_flow_image(flow: torch.Tensor, filepath: str, max_flow: Optional[float] = None):
    """
    Save flow as color image
    
    Args:
        flow: Flow tensor [2, H, W]
        filepath: Path to save image
        max_flow: Maximum flow for normalization
    """
    from PIL import Image
    
    # Visualize
    flow_color = visualize_flow(flow, max_flow)
    
    # Save
    img = Image.fromarray(flow_color)
    img.save(filepath)
    print(f"Saved flow visualization to {filepath}")


def create_flow_legend(max_flow: float = 10.0, size: int = 256) -> np.ndarray:
    """
    Create a color wheel legend for flow visualization
    
    Args:
        max_flow: Maximum flow magnitude
        size: Size of the legend image
    
    Returns:
        Legend image [size, size, 3]
    """
    # Create coordinate grid
    y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    
    # Create circular flow pattern
    flow = np.stack([x, y], axis=-1) * max_flow
    
    # Mask to create circular legend
    radius = np.sqrt(x**2 + y**2)
    mask = radius <= 1.0
    
    # Visualize
    legend = flow_to_color(flow, max_flow)
    
    # Apply mask (white background outside circle)
    legend[~mask] = 255
    
    return legend


def plot_flow_comparison(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                         input_image: Optional[torch.Tensor] = None,
                         save_path: Optional[str] = None):
    """
    Plot comparison between predicted and ground truth flow
    
    Args:
        pred_flow: Predicted flow [2, H, W]
        gt_flow: Ground truth flow [2, H, W]
        input_image: Optional input image [3, H, W]
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3 if input_image is None else 4, figsize=(15, 5))
    
    # Convert flows to color
    max_flow = max(
        torch.abs(gt_flow).max().item(),
        torch.abs(pred_flow).max().item()
    )
    
    pred_vis = visualize_flow(pred_flow, max_flow)
    gt_vis = visualize_flow(gt_flow, max_flow)
    
    # Compute error
    error = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=0))
    error_np = error.detach().cpu().numpy()
    
    # Plot
    idx = 0
    
    if input_image is not None:
        if isinstance(input_image, torch.Tensor):
            input_image = input_image.detach().cpu().numpy()
        
        # Handle different input formats
        if input_image.shape[0] == 3:  # RGB
            input_image = np.transpose(input_image, (1, 2, 0))
            axes[idx].imshow(input_image)
        else:  # Event voxels - show sum
            axes[idx].imshow(input_image.sum(axis=0), cmap='gray')
        
        axes[idx].set_title('Input')
        axes[idx].axis('off')
        idx += 1
    
    axes[idx].imshow(gt_vis)
    axes[idx].set_title('Ground Truth Flow')
    axes[idx].axis('off')
    idx += 1
    
    axes[idx].imshow(pred_vis)
    axes[idx].set_title('Predicted Flow')
    axes[idx].axis('off')
    idx += 1
    
    im = axes[idx].imshow(error_np, cmap='jet')
    axes[idx].set_title(f'Error (EPE: {error.mean().item():.2f})')
    axes[idx].axis('off')
    plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_spike_activity(spike_tensor: torch.Tensor, save_path: Optional[str] = None):
    """
    Visualize spike activity over time
    
    Args:
        spike_tensor: Spike tensor [T, B, C, H, W] or [T, C, H, W]
        save_path: Path to save visualization
    """
    if isinstance(spike_tensor, torch.Tensor):
        spike_tensor = spike_tensor.detach().cpu().numpy()
    
    # Sum over spatial dimensions and channels
    if spike_tensor.ndim == 5:
        activity = spike_tensor.sum(axis=(2, 3, 4))  # [T, B]
        num_plots = min(spike_tensor.shape[1], 4)  # Show up to 4 samples
    else:
        activity = spike_tensor.sum(axis=(1, 2, 3))  # [T]
        num_plots = 1
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3*num_plots))
    if num_plots == 1:
        axes = [axes]
    
    for i in range(num_plots):
        if spike_tensor.ndim == 5:
            axes[i].plot(activity[:, i])
            axes[i].set_title(f'Sample {i} - Spike Activity Over Time')
        else:
            axes[i].plot(activity)
            axes[i].set_title('Spike Activity Over Time')
        
        axes[i].set_xlabel('Timestep')
        axes[i].set_ylabel('Total Spikes')
        axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spike visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
