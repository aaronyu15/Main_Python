"""
Loss functions for optical flow estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from traitlets import Bool


def endpoint_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor, 
                   valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute Average Endpoint Error (AEE)
    
    Args:
        pred_flow: Predicted flow [B, 2, H, W]
        gt_flow: Ground truth flow [B, 2, H, W]
        valid_mask: Valid regions [B, 1, H, W]
    
    Returns:
        Average endpoint error
    """
    error = torch.norm(pred_flow - gt_flow, p=2, dim=1, keepdim=True) 
    
    if valid_mask is not None:
        error = error * valid_mask
        return error.sum() / (valid_mask.sum() + 1e-8)
    else:
        return error.mean().item()

def calculate_effective_epe(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                            valid_mask: Optional[torch.Tensor] = None,
                            threshold: float = 0.1, threshold_max: Optional[float] = None) -> float:
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
    pred_mag = torch.sqrt(torch.sum(pred_flow ** 2, dim=1))
    gt_mag = torch.sqrt(torch.sum(gt_flow ** 2, dim=1))
    
    # Create mask for effective pixels (where flow is significant)
    if threshold_max is not None:
        effective_mask = ((gt_mag > threshold) & (gt_mag < threshold_max)) | ((pred_mag > threshold) & (pred_mag < threshold_max)   )
    else:   
        effective_mask = (gt_mag > threshold) | (pred_mag > threshold)
    
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

  

def angular_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                  valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute Angular Error (AE) between predicted and ground truth flow
    
    Args:
        pred_flow: Predicted flow [B, 2, H, W]
        gt_flow: Ground truth flow [B, 2, H, W]
        valid_mask: Valid regions [B, 1, H, W]
    
    Returns:
        Average angular error
    """
    # Extract u and v components
    pu, pv = pred_flow[:, 0], pred_flow[:, 1]
    gu, gv = gt_flow[:, 0], gt_flow[:, 1]

    # 3D dot: (u,v,1)·(u',v',1) = u*u' + v*v' + 1
    dot = pu * gu + pv * gv + 1.0

    # 3D norms: sqrt(u^2 + v^2 + 1)
    pnorm = torch.sqrt(pu * pu + pv * pv + 1.0)
    gnorm = torch.sqrt(gu * gu + gv * gv + 1.0)

    cos = dot / (pnorm * gnorm + 1e-8)
    cos = torch.clamp(cos, -0.999, 0.999)

    ang = torch.acos(cos) * 180.0 / np.pi  # degrees

    if valid_mask is not None:
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(1)
        valid_mask = valid_mask.to(dtype=ang.dtype)
        return (ang * valid_mask).sum() / (valid_mask.sum() + 1e-8)

    return ang.mean().item()


def multi_scale_endpoint_loss(flow_pyramid: Dict[str, torch.Tensor],
                          gt_flow: torch.Tensor,
                          valid_mask: Optional[torch.Tensor] = None,
                          weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """
    Multi-scale flow loss
    
    Args:
        flow_pyramid: Dictionary of predicted flows at different scales
        gt_flow: Ground truth flow at original resolution [B, 2, H, W]
        valid_mask: Valid regions at original resolution [B, 1, H, W]
        weights: Loss weights for each scale
    
    Returns:
        Weighted multi-scale loss
    """
    if weights is None:
        weights = {
            'flow5': 0.32,
            'flow4': 0.16,
            'flow3': 0.08,
            'flow2': 0.04
        }
    
    total_loss = 0.0
    
    for scale_name, pred_flow in flow_pyramid.items():
        if scale_name not in weights:
            continue
        
        # Downsample ground truth to match prediction scale
        _, _, h, w = pred_flow.shape
        gt_flow_scaled = F.interpolate(gt_flow, size=(h, w), mode='bilinear', align_corners=False)
        
        # Scale the flow values
        scale_factor_h = h / gt_flow.shape[2]
        scale_factor_w = w / gt_flow.shape[3]
        gt_flow_scaled[:, 0] *= scale_factor_w
        gt_flow_scaled[:, 1] *= scale_factor_h
        
        # Downsample valid mask
        if valid_mask is not None:
            valid_mask_scaled = F.interpolate(valid_mask.float(), size=(h, w), mode='nearest')
        else:
            valid_mask_scaled = None
        
        # Compute loss at this scale
        scale_loss = endpoint_error(pred_flow, gt_flow_scaled, valid_mask_scaled, loss_type='robust')
        total_loss += weights[scale_name] * scale_loss
    
    return total_loss


class CombinedLoss(nn.Module):
    """
    Combined loss for SNN optical flow training
    """
    def __init__(
        self,
        endpoint_weight: float = 1.0,
        angular_weight: float = 0.0,
        outlier_weight: float = 1.0,
        effective_epe_weights: Optional[list] = [0.0, 0.0, 0.0, 0.0, 0.0],
    ):
        super().__init__()
        self.endpoint_weight = endpoint_weight
        self.angular_weight = angular_weight
        self.outlier_weight = outlier_weight
        self.eff_endpoint_weight = effective_epe_weights 

    def forward(
        self,
        outputs: Dict,
        gt_flow: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            outputs: Model outputs with 'flow', 'flow_pyramid', 'spike_stats'
            gt_flow: Ground truth flow
            valid_mask: Valid mask
            model: Model for quantization loss
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Main flow loss
        #if 'flow_pyramid' in outputs:
        #    losses['endpoint_loss'] = multi_scale_endpoint_loss(
        #        outputs['flow_pyramid'], gt_flow, valid_mask
        #    )
        #else:
        losses['endpoint_loss'] = endpoint_error(outputs['flow'], gt_flow, valid_mask)
        losses['angular_loss'] = angular_error(outputs['flow'], gt_flow, valid_mask)
        losses['outlier_loss'] = calculate_outliers(outputs['flow'], gt_flow, valid_mask, threshold=1.0)

        losses['endpoint_0p1_loss'] = calculate_effective_epe(outputs['flow'], gt_flow, valid_mask, threshold=0.1, threshold_max=1.0)
        losses['endpoint_1p0_loss'] = calculate_effective_epe(outputs['flow'], gt_flow, valid_mask, threshold=1.0, threshold_max=5.0)
        losses['endpoint_5p0_loss'] = calculate_effective_epe(outputs['flow'], gt_flow, valid_mask, threshold=5.0, threshold_max=20.0)
        losses['endpoint_20p0_loss'] = calculate_effective_epe(outputs['flow'], gt_flow, valid_mask, threshold=20.0, threshold_max=50.0)
        losses['endpoint_50p0_loss'] = calculate_effective_epe(outputs['flow'], gt_flow, valid_mask, threshold=50.0, threshold_max=100.0)
        

        losses['total_loss'] = (
            self.endpoint_weight * losses['endpoint_loss'] +
            self.angular_weight * losses['angular_loss'] +
            self.outlier_weight * losses['outlier_loss'] +
            self.eff_endpoint_weight[0] * losses['endpoint_0p1_loss'] +
            self.eff_endpoint_weight[1] * losses['endpoint_1p0_loss'] +
            self.eff_endpoint_weight[2] * losses['endpoint_5p0_loss'] +
            self.eff_endpoint_weight[3] * losses['endpoint_20p0_loss'] +
            self.eff_endpoint_weight[4] * losses['endpoint_50p0_loss']
        )
        
        return losses

# Metrics
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

    gt_mag = torch.sqrt(torch.sum(gt_flow ** 2, dim=1))

    epe = torch.norm(pred_flow - gt_flow, p=2, dim=1)
    
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