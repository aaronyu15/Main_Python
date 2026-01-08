"""
Metrics for optical flow evaluation
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


def calculate_epe(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                  valid_mask: Optional[torch.Tensor] = None) -> float:
    """
    Calculate Average Endpoint Error (EPE)
    
    Args:
        pred_flow: Predicted flow [B, 2, H, W] or [2, H, W]
        gt_flow: Ground truth flow [B, 2, H, W] or [2, H, W]
        valid_mask: Valid mask [B, 1, H, W] or [1, H, W]
    
    Returns:
        Average endpoint error
    """
    # Compute L2 error
    error = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=-3 if pred_flow.dim() == 4 else 0))
    
    if valid_mask is not None:
        valid_mask = valid_mask.squeeze(-3 if valid_mask.dim() == 4 else 0)
        error = error * valid_mask
        return (error.sum() / (valid_mask.sum() + 1e-8)).item()
    else:
        return error.mean().item()


def calculate_outliers(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                       valid_mask: Optional[torch.Tensor] = None,
                       threshold: float = 3.0) -> float:
    """
    Calculate percentage of outlier pixels
    
    An outlier is defined as EPE > threshold or EPE > 5% of ground truth magnitude
    
    Args:
        pred_flow: Predicted flow [B, 2, H, W] or [2, H, W]
        gt_flow: Ground truth flow [B, 2, H, W] or [2, H, W]
        valid_mask: Valid mask
        threshold: Absolute threshold
    
    Returns:
        Percentage of outliers
    """
    # Compute endpoint error
    epe = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=-3 if pred_flow.dim() == 4 else 0))
    
    # Ground truth magnitude
    gt_mag = torch.sqrt(torch.sum(gt_flow ** 2, dim=-3 if gt_flow.dim() == 4 else 0))
    
    # Outlier mask: EPE > threshold AND EPE > 5% of GT magnitude
    outliers = (epe > threshold) & (epe > 0.05 * gt_mag)
    
    if valid_mask is not None:
        valid_mask = valid_mask.squeeze(-3 if valid_mask.dim() == 4 else 0)
        outliers = outliers * valid_mask
        return (outliers.sum() / (valid_mask.sum() + 1e-8) * 100).item()
    else:
        return (outliers.float().mean() * 100).item()


def calculate_angular_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                           valid_mask: Optional[torch.Tensor] = None) -> float:
    """
    Calculate average angular error
    
    Args:
        pred_flow: Predicted flow [B, 2, H, W] or [2, H, W]
        gt_flow: Ground truth flow [B, 2, H, W] or [2, H, W]
        valid_mask: Valid mask
    
    Returns:
        Average angular error in degrees
    """
    # Add dimension for 3D vectors [u, v, 1]
    if pred_flow.dim() == 3:
        pred_flow = pred_flow.unsqueeze(0)
        gt_flow = gt_flow.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size = pred_flow.shape[0]
    
    # Create 3D vectors
    pred_vec = torch.cat([
        pred_flow,
        torch.ones(batch_size, 1, pred_flow.shape[2], pred_flow.shape[3], device=pred_flow.device)
    ], dim=1)
    
    gt_vec = torch.cat([
        gt_flow,
        torch.ones(batch_size, 1, gt_flow.shape[2], gt_flow.shape[3], device=gt_flow.device)
    ], dim=1)
    
    # Normalize
    pred_vec = pred_vec / (torch.norm(pred_vec, dim=1, keepdim=True) + 1e-8)
    gt_vec = gt_vec / (torch.norm(gt_vec, dim=1, keepdim=True) + 1e-8)
    
    # Dot product
    dot_product = torch.sum(pred_vec * gt_vec, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Angular error
    angular_error = torch.acos(dot_product) * 180.0 / np.pi
    
    if valid_mask is not None:
        valid_mask = valid_mask.squeeze(1)
        angular_error = angular_error * valid_mask
        return (angular_error.sum() / (valid_mask.sum() + 1e-8)).item()
    else:
        return angular_error.mean().item()


def compute_metrics(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                   valid_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute all metrics
    
    Args:
        pred_flow: Predicted flow
        gt_flow: Ground truth flow
        valid_mask: Valid mask
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'epe': calculate_epe(pred_flow, gt_flow, valid_mask),
        'outliers': calculate_outliers(pred_flow, gt_flow, valid_mask, threshold=3.0),
        'angular_error': calculate_angular_error(pred_flow, gt_flow, valid_mask)
    }
    
    return metrics


def compute_metrics_by_magnitude(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                                 valid_mask: Optional[torch.Tensor] = None,
                                 bins: list = [0, 10, 40, float('inf')]) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics stratified by flow magnitude
    
    Args:
        pred_flow: Predicted flow
        gt_flow: Ground truth flow
        valid_mask: Valid mask
        bins: Magnitude bins
    
    Returns:
        Dictionary of metrics for each bin
    """
    # Compute ground truth magnitude
    gt_mag = torch.sqrt(torch.sum(gt_flow ** 2, dim=-3 if gt_flow.dim() == 4 else 0))
    
    results = {}
    
    for i in range(len(bins) - 1):
        bin_min, bin_max = bins[i], bins[i+1]
        
        # Create mask for this magnitude bin
        bin_mask = (gt_mag >= bin_min) & (gt_mag < bin_max)
        
        if valid_mask is not None:
            bin_mask = bin_mask & valid_mask.squeeze(-3 if valid_mask.dim() == 4 else 0).bool()
        
        bin_mask = bin_mask.unsqueeze(-3 if pred_flow.dim() == 4 else 0).float()
        
        # Compute metrics for this bin
        if bin_mask.sum() > 0:
            bin_name = f'{bin_min}-{bin_max}' if bin_max != float('inf') else f'{bin_min}+'
            results[bin_name] = compute_metrics(pred_flow, gt_flow, bin_mask)
        
    return results
