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
    # Compute L2 distance
    error = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=1, keepdim=True))
    
    if valid_mask is not None:
        error = error * valid_mask
        return error.sum() / (valid_mask.sum() + 1e-8)
    else:
        return error.mean()


def angular_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                  valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute Angular Error (AE) between predicted and ground truth flow
    
    Angular error measures the angle between the 3D vectors (u, v, 1) formed by
    the flow fields. It's a more perceptually meaningful metric than endpoint error
    as it considers flow direction, not just magnitude.
    
    Args:
        pred_flow: Predicted flow [B, 2, H, W]
        gt_flow: Ground truth flow [B, 2, H, W]
        valid_mask: Valid regions [B, 1, H, W]
    
    Returns:
        Average angular error in degrees
    """
    # Extract u and v components
    pred_u, pred_v = pred_flow[:, 0], pred_flow[:, 1]  # [B, H, W]
    gt_u, gt_v = gt_flow[:, 0], gt_flow[:, 1]  # [B, H, W]
    
    # Compute 3D vector magnitudes: sqrt(u^2 + v^2 + 1)
    pred_norm = torch.sqrt(pred_u**2 + pred_v**2 + 1.0)
    gt_norm = torch.sqrt(gt_u**2 + gt_v**2 + 1.0)
    
    # Compute dot product: u1*u2 + v1*v2 + 1*1
    dot_product = pred_u * gt_u + pred_v * gt_v + 1.0
    
    # Use numerically stable formulation with atan2 instead of acos
    # angle = atan2(||a x b||, a · b)
    # For 3D vectors (u1, v1, 1) and (u2, v2, 1):
    # Cross product magnitude in the plane perpendicular to z:
    # ||cross|| = sqrt((v1 - v2)^2 + (u2 - u1)^2 + (u1*v2 - u2*v1)^2)
    # But we can use a simpler numerically stable form:
    
    # Compute cosine and sine components safely
    cos_angle = dot_product / (pred_norm * gt_norm + 1e-8)
    
    # For numerical stability, use atan2 formulation
    # sin_angle = ||cross_product|| / (norm1 * norm2)
    # cross_product for (u1,v1,1) x (u2,v2,1) has magnitude:
    cross_u = pred_v - gt_v  # from the 1*v2 - 1*v1 component
    cross_v = gt_u - pred_u  # from the u1*1 - u2*1 component
    cross_z = pred_u * gt_v - pred_v * gt_u  # standard 2D cross product
    cross_magnitude = torch.sqrt(cross_u**2 + cross_v**2 + cross_z**2 + 1e-8)
    
    sin_angle = cross_magnitude / (pred_norm * gt_norm + 1e-8)
    
    # Use atan2 for numerical stability (avoids issues at boundaries)
    angle = torch.atan2(sin_angle, cos_angle)
    angle_deg = angle * 180.0 / torch.pi
    
    if valid_mask is not None:
        # Squeeze valid_mask if it has channel dimension
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(1)  # [B, H, W]
        angle_deg = angle_deg * valid_mask
        return angle_deg.sum() / (valid_mask.sum() + 1e-8)
    else:
        return angle_deg.mean()


def flow_loss(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
              valid_mask: Optional[torch.Tensor] = None,
              loss_type: str = 'l1') -> torch.Tensor:
    """
    Basic flow loss
    
    Args:
        pred_flow: Predicted flow [B, 2, H, W]
        gt_flow: Ground truth flow [B, 2, H, W]
        valid_mask: Valid regions [B, 1, H, W]
        loss_type: 'l1', 'l2', or 'robust'
    
    Returns:
        Loss value
    """
    if loss_type == 'l1':
        loss = torch.abs(pred_flow - gt_flow)
    elif loss_type == 'l2':
        loss = (pred_flow - gt_flow) ** 2
    elif loss_type == 'robust':
        # Robust loss (Charbonnier)
        epsilon = 0.01
        loss = torch.sqrt((pred_flow - gt_flow) ** 2 + epsilon ** 2)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    if valid_mask is not None:
        loss = loss * valid_mask
        return loss.sum() / (valid_mask.sum() * 2 + 1e-8)  # Divide by 2 for u,v channels
    else:
        return loss.mean()


def multi_scale_flow_loss(flow_pyramid: Dict[str, torch.Tensor],
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
        scale_loss = flow_loss(pred_flow, gt_flow_scaled, valid_mask_scaled, loss_type='robust')
        total_loss += weights[scale_name] * scale_loss
    
    return total_loss





class CombinedLoss(nn.Module):
    """
    Combined loss for SNN optical flow training
    """
    def __init__(
        self,
        flow_weight: float = 1.0,
        angular_weight: float = 0.0,
    ):
        super().__init__()
        self.flow_weight = flow_weight
        self.angular_weight = angular_weight

        
    def forward(
        self,
        outputs: Dict,
        gt_flow: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None
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
            losses['flow_loss'] = multi_scale_flow_loss(
                outputs['flow_pyramid'], gt_flow, valid_mask
            )
        else:
            losses['flow_loss'] = flow_loss(outputs['flow'], gt_flow, valid_mask)
        
        # Angular error loss
        losses['angular_loss'] = angular_error(outputs['flow'], gt_flow, valid_mask)

        # Total loss
        losses['total_loss'] = (
            self.flow_weight * losses['flow_loss'] +
            self.angular_weight * losses['angular_loss']
        )
        
        return losses
