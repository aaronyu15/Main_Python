import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import sys
import traceback

from traitlets import Bool

def epe (pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
    # [B, 2, H, W] -> [B, 1, H, W]
    return torch.norm(pred_flow - gt_flow, p=2, dim=1, keepdim=True)

def log_epe (pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
    # [B, 2, H, W] -> [B, 1, H, W]
    return torch.log1p(torch.norm(pred_flow - gt_flow, p=2, dim=1, keepdim=True))

def apply_mask(flow: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        if mask.shape != flow.shape:
            print(f"Mask shape {mask.shape} does not match flow shape {flow.shape}")
            traceback.print_stack()
            sys.exit(1)
        return flow * mask, mask
    else:
        mask = torch.ones_like(flow)
        return flow, mask


def endpoint_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None, return_vec: bool = False, proc="log", mode="gen") -> torch.Tensor:
    if mode == "gen":
        if proc == "log":
            error = log_epe(pred_flow, gt_flow)
        elif proc == "epe":
            error = epe(pred_flow, gt_flow)

        error, mask = apply_mask(error, mask)
    
        if return_vec:
            return error, mask
        else:
            return error.sum() / (mask.sum() + 1e-8)

    elif mode == "directional":
        # Compute per-pixel EPE then aggregate by GT direction quadrants (90° FOV)
        if proc == "log":
            error_map = log_epe(pred_flow, gt_flow)  # [B,1,H,W]
        else:
            error_map = epe(pred_flow, gt_flow)      # [B,1,H,W]

        base_mask = mask if mask is not None else torch.ones_like(error_map)

        gt_u, gt_v = gt_flow[:, 0], gt_flow[:, 1]
        ang_deg = torch.atan2(gt_v, gt_u) * 180.0 / np.pi
        ang_deg = (ang_deg + 360.0) % 360.0  # [0, 360)

        dir_masks = {
            "right": ((ang_deg >= 315.0) | (ang_deg < 45.0)).unsqueeze(1).float(),
            "up":    ((ang_deg >= 45.0)  & (ang_deg < 135.0)).unsqueeze(1).float(),
            "left":  ((ang_deg >= 135.0) & (ang_deg < 225.0)).unsqueeze(1).float(),
            "down":  ((ang_deg >= 225.0) & (ang_deg < 315.0)).unsqueeze(1).float(),
        }

        errors = {}
        for name, dir_mask in dir_masks.items():
            combined_mask = dir_mask * base_mask
            errors[name] = (error_map * combined_mask).sum() / (combined_mask.sum() + 1e-8)

        return errors


def effective_epe(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  threshold_min: float = 0.1,
                  threshold_max: Optional[float] = None, proc="log") -> float:

    pred_mag = torch.sqrt(torch.sum(pred_flow ** 2, dim=1, keepdim=True))
    gt_mag = torch.sqrt(torch.sum(gt_flow ** 2, dim=1, keepdim=True))

    threshold_max = threshold_max if threshold_max is not None else float('inf')
    
    effective_mask = ((gt_mag > threshold_min) & (gt_mag < threshold_max)) | ((pred_mag > threshold_min) & (pred_mag < threshold_max))
    
    mask = mask * effective_mask
    
    return endpoint_error(pred_flow, gt_flow, mask, proc=proc)
  

def angular_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                  mask: Optional[torch.Tensor] = None, return_vec: bool = False, mode: str = "gen") -> torch.Tensor:
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
    ang = ang.unsqueeze(1)  # [B, 1, H, W]

    nonzero_mask = (gt_flow.norm(dim=1) > 0.1).float().unsqueeze(1)  # [B, 1, H, W]
    if mode == "gen":
        mask = nonzero_mask if mask is None else (mask * nonzero_mask)
        ang, mask = apply_mask(ang, mask)

        if return_vec:
            return ang, mask
        return ang.sum() / (mask.sum() + 1e-8)

    elif mode == "directional":
        base_mask = mask if mask is not None else torch.ones_like(ang)

        ang_deg = torch.atan2(gv, gu) * 180.0 / np.pi
        ang_deg = (ang_deg + 360.0) % 360.0

        dir_masks = {
            "right": ((ang_deg >= 315.0) | (ang_deg < 45.0)).unsqueeze(1).float() * nonzero_mask,
            "up":    ((ang_deg >= 45.0)  & (ang_deg < 135.0)).unsqueeze(1).float() * nonzero_mask,
            "left":  ((ang_deg >= 135.0) & (ang_deg < 225.0)).unsqueeze(1).float() * nonzero_mask,
            "down":  ((ang_deg >= 225.0) & (ang_deg < 315.0)).unsqueeze(1).float() * nonzero_mask,
        }

        errors = {}
        for name, dir_mask in dir_masks.items():
            combined_mask = dir_mask * base_mask
            errors[name] = (ang * combined_mask).sum() / (combined_mask.sum() + 1e-8)

        if return_vec:
            return errors, dir_masks
        return errors

def epe_weighted_angular_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor, inputs: torch.Tensor,
                              mask: Optional[torch.Tensor] = None, proc="log") -> torch.Tensor:
    """
    Compute angular error weighted by EPE magnitude and event activity.
    Regions with more events and larger errors contribute more to the loss.
    """
    ang_error, ang_mask = angular_error(pred_flow, gt_flow, mask, return_vec=True)
    epe_error, epe_mask = endpoint_error(pred_flow, gt_flow, mask, return_vec=True, proc=proc)

    combined_mask = ang_mask * epe_mask 
    
    # Sum event activity over time and polarity bins to get [B, 1, H, W]
    event_activity = inputs.sum(dim=1).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    
    # Normalize event activity to [0, 1] range per sample to prevent extreme weighting
    # This ensures samples with different overall event counts are treated fairly
    max_activity = event_activity.view(event_activity.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
    event_weight = event_activity / (max_activity + 1e-8)
    
    # Combine weights: angular error * EPE error * event activity
    weighted_ang_error = ang_error * epe_error * event_weight 

    # Normalize by sum of weights (not just mask count) to account for variable event activity
    total_weight = (event_weight * combined_mask).sum()
    
    return weighted_ang_error.sum() / (total_weight + 1e-8)

def null_prediction_loss(pred_flow: torch.Tensor, gt_flow: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

    gt_mag = torch.sqrt(torch.sum(gt_flow ** 2, dim=1, keepdim=True))
    zero_gt_mask = (gt_mag < 0.1).float()  # [B, 1, H, W]

    null_pred_flow = pred_flow * zero_gt_mask  # [B, 2, H, W]

    loss = torch.norm(null_pred_flow, p=2, dim=1, keepdim=True)  # [B, 1, H, W]

    return loss.sum() / (zero_gt_mask.sum() + 1e-8)

def smoothness_loss(flow: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Extract u and v components
    pu, pv = flow[:, 0].unsqueeze(1), flow[:, 1].unsqueeze(1)
    
    # Shift to get neighboring pixels (replicate padding at borders)
    # For 4D tensors, padding format is (left, right, top, bottom)
    pu_right = F.pad(pu[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')
    pu_down = F.pad(pu[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')
    pv_right = F.pad(pv[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')
    pv_down = F.pad(pv[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')

    # Compute 3D dot products: (u,v,1)·(u',v',1)
    dot_right = pu * pu_right + pv * pv_right + 1.0
    dot_down = pu * pu_down + pv * pv_down + 1.0

    # Compute 3D norms: sqrt(u^2 + v^2 + 1)
    pnorm = torch.sqrt(pu * pu + pv * pv + 1.0)
    p_right_norm = torch.sqrt(pu_right * pu_right + pv_right * pv_right + 1.0)
    p_down_norm = torch.sqrt(pu_down * pu_down + pv_down * pv_down + 1.0)

    # Compute angular differences
    cos_right = dot_right / (pnorm * p_right_norm + 1e-8)
    cos_right = torch.clamp(cos_right, -0.999, 0.999)
    ang_right = torch.acos(cos_right) * 180.0 / np.pi  # degrees
    ang_right_error = ang_right > 10.0

    cos_down = dot_down / (pnorm * p_down_norm + 1e-8)
    cos_down = torch.clamp(cos_down, -0.999, 0.999)
    ang_down = torch.acos(cos_down) * 180.0 / np.pi  # degrees
    ang_down_error = ang_down > 10.0

    ang_right, mask1 = apply_mask(ang_right, ang_right_error)
    ang_down, mask2 = apply_mask(ang_down, ang_down_error)

    mask = mask1 | mask2

    return (ang_right + ang_down).sum() / (mask.sum() + 1e-8)


def vertical_loss(pred_flow: torch.Tensor, gt_flow: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

    pred_v = pred_flow[:, 1].unsqueeze(1)  # [B, 1, H, W]
    gt_v = gt_flow[:, 1].unsqueeze(1)  # [B, 1, H, W]
    
    v_error = torch.abs(pred_v - gt_v)
    
    loss, mask = apply_mask(v_error, mask)

    return loss.sum() / (mask.sum() + 1e-8)


class CombinedLoss(nn.Module):
    """
    Combined loss for SNN optical flow training
    """
    def __init__(
        self,
        endpoint_weight: float = 1.0,
        angular_weight: float = 0.0,
        epe_ang_weight: float = 0.0,
        smoothness_weight: float = 1.0,
        vertical_weight: float = 0.0,
        null_pred_weight: float = 0.0,
        effective_epe_weights: Optional[list] = [0.0, 0.0, 0.0, 0.0, 0.0],
        epe_type = "log",
        directional_balance_weight: float = 0.0,
    ):
        super().__init__()
        self.endpoint_weight = endpoint_weight
        self.angular_weight = angular_weight
        self.epe_ang_weight = epe_ang_weight

        self.vertical_weight = vertical_weight
        self.smoothness_weight = smoothness_weight
        self.null_pred_weight = null_pred_weight

        self.eff_endpoint_weight = effective_epe_weights 

        self.epe_type = epe_type

        # Directional balance regularizer (variance of directional angular errors)
        self.directional_balance_weight = directional_balance_weight

    def forward(
        self,
        outputs: Dict,
        gt_flow: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            outputs: Model outputs with 'flow', 'flow_pyramid', 'spike_stats'
            gt_flow: Ground truth flow
            mask: Valid mask
            model: Model for quantization loss
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        losses['endpoint_loss'] = endpoint_error(outputs['flow'], gt_flow, mask, proc=self.epe_type)
        losses['angular_loss'] = angular_error(outputs['flow'], gt_flow, mask)
        losses['epe_ang_loss'] = epe_weighted_angular_error(outputs['flow'], gt_flow, inputs, mask, proc=self.epe_type)

        losses['smoothness_loss'] = smoothness_loss(outputs['flow'], mask)
        losses['null_pred_loss'] = null_prediction_loss(outputs['flow'], gt_flow, mask)

        losses['endpoint_0p1_loss'] = effective_epe(outputs['flow'], gt_flow, mask, threshold_min=0.1, threshold_max=1.0, proc=self.epe_type)
        losses['endpoint_1p0_loss'] = effective_epe(outputs['flow'], gt_flow, mask, threshold_min=1.0, threshold_max=5.0, proc=self.epe_type)
        losses['endpoint_5p0_loss'] = effective_epe(outputs['flow'], gt_flow, mask, threshold_min=5.0, threshold_max=20.0, proc=self.epe_type)
        losses['endpoint_20p0_loss'] =effective_epe(outputs['flow'], gt_flow, mask, threshold_min=20.0, proc=self.epe_type)

        # Directional balance on angular error (variance across quadrants)
        if self.directional_balance_weight > 0.0:
            dir_ang = angular_error(
                outputs['flow'],
                gt_flow,
                mask,
                mode="directional"
            )

            # Compute variance separately for horizontal (L/R) and vertical (U/D), then average
            ang_left = dir_ang['left'] if torch.is_tensor(dir_ang['left']) else torch.tensor(dir_ang['left'], device=gt_flow.device)
            ang_right = dir_ang['right'] if torch.is_tensor(dir_ang['right']) else torch.tensor(dir_ang['right'], device=gt_flow.device)
            ang_up = dir_ang['up'] if torch.is_tensor(dir_ang['up']) else torch.tensor(dir_ang['up'], device=gt_flow.device)
            ang_down = dir_ang['down'] if torch.is_tensor(dir_ang['down']) else torch.tensor(dir_ang['down'], device=gt_flow.device)

            horiz = torch.stack([ang_left, ang_right])
            vert = torch.stack([ang_up, ang_down])

            var_horiz = ((horiz - horiz.mean()) ** 2).mean()
            var_vert = ((vert - vert.mean()) ** 2).mean()

            dir_balance = 0.5 * (var_horiz + var_vert)
            losses['dir_balance_loss'] = dir_balance * self.directional_balance_weight
        else:
            losses['dir_balance_loss'] = torch.tensor(0.0, device=gt_flow.device)
        

        losses['total_loss'] = (
            self.endpoint_weight * losses['endpoint_loss'] +
            self.angular_weight * losses['angular_loss'] +
            self.epe_ang_weight * losses['epe_ang_loss'] +
            self.smoothness_weight * losses['smoothness_loss'] +
            self.null_pred_weight * losses['null_pred_loss'] +
            self.eff_endpoint_weight[0] * losses['endpoint_0p1_loss'] +
            self.eff_endpoint_weight[1] * losses['endpoint_1p0_loss'] +
            self.eff_endpoint_weight[2] * losses['endpoint_5p0_loss'] +
            self.eff_endpoint_weight[3] * losses['endpoint_20p0_loss'] +
            losses['dir_balance_loss']
        )
        
        return losses

# Metrics
def calculate_outliers(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       threshold: float = 3.0) -> float:

    gt_mag = torch.sqrt(torch.sum(gt_flow ** 2, dim=1, keepdim=True))

    epe = torch.norm(pred_flow - gt_flow, p=2, dim=1, keepdim=True)
    
    outliers = (epe > threshold) & (epe > 0.05 * gt_mag)

    outliers, mask = apply_mask(outliers.float(), mask)

    return (outliers.sum() / (mask.sum() + 1e-8) * 100).item()



