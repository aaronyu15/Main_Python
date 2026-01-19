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


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    """
    Smoothness loss to encourage locally smooth flow fields
    
    Args:
        flow: Flow tensor [B, 2, H, W]
    
    Returns:
        Smoothness loss
    """
    # Compute gradients
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    
    # L1 smoothness
    smooth_loss = (torch.abs(dx).mean() + torch.abs(dy).mean()) / 2.0
    
    return smooth_loss


def photometric_loss(img1: torch.Tensor, img2_warped: torch.Tensor,
                     valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Photometric loss for self-supervised learning
    
    Args:
        img1: First image [B, C, H, W]
        img2_warped: Second image warped by predicted flow [B, C, H, W]
        valid_mask: Valid regions [B, 1, H, W]
    
    Returns:
        Photometric loss
    """
    # Robust photometric loss
    epsilon = 0.01
    diff = torch.sqrt((img1 - img2_warped) ** 2 + epsilon ** 2)
    
    if valid_mask is not None:
        diff = diff * valid_mask
        return diff.sum() / (valid_mask.sum() * img1.shape[1] + 1e-8)
    else:
        return diff.mean()


class SparsityLoss(nn.Module):
    """
    Sparsity loss for SNNs to encourage efficient spiking
    Important for hardware deployment to reduce power consumption
    """
    def __init__(self, target_rate: float = 0.1):
        """
        Args:
            target_rate: Target spike rate (0.0 to 1.0)
        """
        super().__init__()
        self.target_rate = target_rate
    
    def forward(self, spike_stats: Dict) -> torch.Tensor:
        """
        Compute sparsity loss from spike statistics
        
        Args:
            spike_stats: Dictionary with 'spike_rate' key
        
        Returns:
            Sparsity loss
        """
        actual_rate = spike_stats.get('spike_rate', 0.0)
        
        # L2 loss between actual and target rate
        loss = (actual_rate - self.target_rate) ** 2
        
        return loss


class QuantizationLoss(nn.Module):
    """
    Regularization loss to encourage weight distributions
    suitable for quantization
    """
    def __init__(self, weight_decay: float = 1e-4):
        super().__init__()
        self.weight_decay = weight_decay
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute quantization-friendly regularization
        
        Args:
            model: Neural network model
        
        Returns:
            Regularization loss
        """
        loss = 0.0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                # L2 regularization
                loss += torch.sum(param ** 2)
        
        return self.weight_decay * loss


class CombinedLoss(nn.Module):
    """
    Combined loss for SNN optical flow training
    """
    def __init__(
        self,
        flow_weight: float = 1.0,
        smooth_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        quant_weight: float = 0.0001,
        target_spike_rate: float = 0.1
    ):
        super().__init__()
        self.flow_weight = flow_weight
        self.smooth_weight = smooth_weight
        self.sparsity_weight = sparsity_weight
        self.quant_weight = quant_weight
        
        self.sparsity_loss = SparsityLoss(target_spike_rate)
        self.quant_loss = QuantizationLoss()
    
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
        
        # Smoothness loss
        losses['smooth_loss'] = smoothness_loss(outputs['flow'])
        
        # Sparsity loss (for SNNs)
        if 'spike_stats' in outputs:
            losses['sparsity_loss'] = self.sparsity_loss(outputs['spike_stats'])
        else:
            losses['sparsity_loss'] = torch.tensor(0.0, device=gt_flow.device)
        
        # Quantization regularization
        if model is not None:
            losses['quant_loss'] = self.quant_loss(model)
        else:
            losses['quant_loss'] = torch.tensor(0.0, device=gt_flow.device)
        
        # Total loss
        losses['total_loss'] = (
            self.flow_weight * losses['flow_loss']
            #self.smooth_weight * losses['smooth_loss'] +
            #self.sparsity_weight * losses['sparsity_loss']
            #self.quant_weight * losses['quant_loss']
        )
        
        return losses
