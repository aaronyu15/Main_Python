"""
Metrics for optical flow evaluation
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from ..training.losses import angular_error, endpoint_error


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
    epe = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=1))
    
    # Ground truth magnitude
    gt_mag = torch.sqrt(torch.sum(gt_flow ** 2, dim=1 ))
    
    # Outlier mask: EPE > threshold AND EPE > 5% of GT magnitude
    outliers = (epe > threshold) & (epe > 0.05 * gt_mag)
    
    if valid_mask is not None:
        valid_mask = valid_mask.squeeze(1)
        outliers = outliers * valid_mask
        return (outliers.sum() / (valid_mask.sum() + 1e-8) * 100).item()
    else:
        return (outliers.float().mean() * 100).item()


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
        'epe': endpoint_error(pred_flow, gt_flow, valid_mask),
        'outliers': calculate_outliers(pred_flow, gt_flow, valid_mask, threshold=3.0),
        'angular_error': angular_error(pred_flow, gt_flow, valid_mask)
    }
    
    return metrics




def calculate_effective_epe(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                            valid_mask: Optional[torch.Tensor] = None,
                            threshold: float = 0.1) -> float:
    """
    Calculate EPE for pixels with ground truth flow magnitude above threshold.
    This focuses on moving objects rather than static regions.
    
    Args:
        pred_flow: Predicted flow [B, 2, H, W] or [2, H, W]
        gt_flow: Ground truth flow [B, 2, H, W] or [2, H, W]
        valid_mask: Valid mask [B, 1, H, W] or [1, H, W]
        threshold: Minimum GT flow magnitude to consider (default: 0.1)
    
    Returns:
        EPE for effective pixels only
    """
    # Compute GT flow magnitude
    gt_mag = torch.sqrt(torch.sum(gt_flow ** 2, dim=1))
    
    # Create mask for effective pixels (where flow is significant)
    effective_mask = (gt_mag > threshold)
    
    if valid_mask is not None:
        valid_mask_2d = valid_mask.squeeze(1)
        effective_mask = effective_mask & valid_mask_2d.bool()
    
    # If no effective pixels, return 0
    if effective_mask.sum() == 0:
        return 0.0
    
    # Expand mask to match flow dimensions
    effective_mask = effective_mask.unsqueeze(1).float()
    
    # Compute EPE only for effective pixels
    return endpoint_error(pred_flow, gt_flow, effective_mask)


def calculate_percentile_epe(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                             valid_mask: Optional[torch.Tensor] = None,
                             percentile: float = 50.0) -> Dict[str, float]:
    """
    Calculate EPE for top percentile of GT flow magnitudes.
    Shows model performance on the most dynamic regions.
    
    Args:
        pred_flow: Predicted flow [B, 2, H, W] or [2, H, W]
        gt_flow: Ground truth flow [B, 2, H, W] or [2, H, W]
        valid_mask: Valid mask [B, 1, H, W] or [1, H, W]
        percentile: Percentile threshold (e.g., 50 = top 50%)
    
    Returns:
        Dictionary with EPE and threshold value
    """
    # Compute GT flow magnitude
    gt_mag = torch.sqrt(torch.sum(gt_flow ** 2, dim=1))
    
    # Get valid pixels
    if valid_mask is not None:
        valid_mask_2d = valid_mask.squeeze(1)
        valid_gt_mag = gt_mag[valid_mask_2d.bool()]
    else:
        valid_gt_mag = gt_mag.flatten()
    
    # If no valid pixels, return 0
    if valid_gt_mag.numel() == 0:
        return {'epe': 0.0, 'threshold': 0.0, 'num_pixels': 0}
    
    # Calculate percentile threshold
    threshold = torch.quantile(valid_gt_mag, percentile / 100.0).item()
    
    # Create mask for pixels above threshold
    percentile_mask = (gt_mag >= threshold)
    
    if valid_mask is not None:
        percentile_mask = percentile_mask & valid_mask_2d.bool()
    
    # Expand mask to match flow dimensions
    percentile_mask = percentile_mask.unsqueeze(1).float()
    
    # Compute EPE for these pixels
    epe = endpoint_error(pred_flow, gt_flow, percentile_mask)
    num_pixels = int(percentile_mask.sum().item())
    
    return {
        'epe': epe,
        'threshold': threshold,
        'num_pixels': num_pixels
    }


def calculate_multi_percentile_epe(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                                   valid_mask: Optional[torch.Tensor] = None,
                                   percentiles: list = [50, 75, 90, 95]) -> Dict[str, float]:
    """
    Calculate EPE for multiple percentiles of flow magnitude.
    Useful for understanding model performance across different motion scales.
    
    Args:
        pred_flow: Predicted flow [B, 2, H, W] or [2, H, W]
        gt_flow: Ground truth flow [B, 2, H, W] or [2, H, W]
        valid_mask: Valid mask [B, 1, H, W] or [1, H, W]
        percentiles: List of percentiles to compute (e.g., [50, 75, 90, 95])
    
    Returns:
        Dictionary of EPE values for each percentile
    """
    results = {}
    
    for p in percentiles:
        p_result = calculate_percentile_epe(pred_flow, gt_flow, valid_mask, p)
        results[f'epe_top{int(100-p)}pct'] = p_result['epe']
        results[f'threshold_top{int(100-p)}pct'] = p_result['threshold']
    
    return results
