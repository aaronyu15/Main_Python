"""
Loss functions for optical flow estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


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
    # Compute L2 distance with numerical stability
    error = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=1, keepdim=True) + 1e-8)
    
    if valid_mask is not None:
        error = error * valid_mask
        return error.sum() / (valid_mask.sum() + 1e-8)
    else:
        return error.mean().item()


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

    ang = torch.acos(cos)  # radians

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
    ):
        super().__init__()
        self.endpoint_weight = endpoint_weight
        self.angular_weight = angular_weight

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
        if 'flow_pyramid' in outputs:
            losses['endpoint_loss'] = multi_scale_endpoint_loss(
                outputs['flow_pyramid'], gt_flow, valid_mask
            )
        else:
            losses['endpoint_loss'] = endpoint_error(outputs['flow'], gt_flow, valid_mask)
        
        losses['angular_loss'] = angular_error(outputs['flow'], gt_flow, valid_mask)

        losses['total_loss'] = (
            self.endpoint_weight * losses['endpoint_loss'] +
            self.angular_weight * losses['angular_loss']
        )
        
        return losses
