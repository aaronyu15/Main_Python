"""
Distillation losses for knowledge transfer from teacher to student models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


def flow_mse_loss(student_flow: torch.Tensor, teacher_flow: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    MSE loss between student and teacher flow predictions.
    
    Args:
        student_flow: Student model flow prediction [B, 2, H, W]
        teacher_flow: Teacher model flow prediction [B, 2, H, W]
        mask: Optional valid mask [B, 1, H, W]
    
    Returns:
        Scalar MSE loss
    """
    mse = (student_flow - teacher_flow) ** 2  # [B, 2, H, W]
    mse = mse.sum(dim=1, keepdim=True)  # [B, 1, H, W]
    
    if mask is not None:
        mse = mse * mask
        return mse.sum() / (mask.sum() * 2 + 1e-8)
    else:
        return mse.mean()


def flow_cosine_loss(student_flow: torch.Tensor, teacher_flow: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Cosine similarity loss between student and teacher flow predictions.
    Focuses on direction matching rather than magnitude.
    
    Args:
        student_flow: Student model flow prediction [B, 2, H, W]
        teacher_flow: Teacher model flow prediction [B, 2, H, W]
        mask: Optional valid mask [B, 1, H, W]
    
    Returns:
        Scalar cosine loss (1 - cosine_similarity)
    """
    # Normalize flow vectors
    student_norm = F.normalize(student_flow, p=2, dim=1)  # [B, 2, H, W]
    teacher_norm = F.normalize(teacher_flow, p=2, dim=1)  # [B, 2, H, W]
    
    # Cosine similarity per pixel
    cos_sim = (student_norm * teacher_norm).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    
    # Convert to loss (1 - similarity)
    cos_loss = 1.0 - cos_sim
    
    if mask is not None:
        cos_loss = cos_loss * mask
        return cos_loss.sum() / (mask.sum() + 1e-8)
    else:
        return cos_loss.mean()


def flow_smooth_l1_loss(student_flow: torch.Tensor, teacher_flow: torch.Tensor,
                        mask: Optional[torch.Tensor] = None,
                        beta: float = 1.0) -> torch.Tensor:
    """
    Smooth L1 (Huber) loss between student and teacher flow predictions.
    More robust to outliers than MSE.
    
    Args:
        student_flow: Student model flow prediction [B, 2, H, W]
        teacher_flow: Teacher model flow prediction [B, 2, H, W]
        mask: Optional valid mask [B, 1, H, W]
        beta: Threshold for switching between L1 and L2
    
    Returns:
        Scalar smooth L1 loss
    """
    loss = F.smooth_l1_loss(student_flow, teacher_flow, reduction='none', beta=beta)
    loss = loss.sum(dim=1, keepdim=True)  # [B, 1, H, W]
    
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() * 2 + 1e-8)
    else:
        return loss.mean()


def flow_epe_loss(student_flow: torch.Tensor, teacher_flow: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Endpoint error loss between student and teacher flow predictions.
    
    Args:
        student_flow: Student model flow prediction [B, 2, H, W]
        teacher_flow: Teacher model flow prediction [B, 2, H, W]
        mask: Optional valid mask [B, 1, H, W]
    
    Returns:
        Scalar EPE loss
    """
    epe = torch.norm(student_flow - teacher_flow, p=2, dim=1, keepdim=True)  # [B, 1, H, W]
    
    if mask is not None:
        epe = epe * mask
        return epe.sum() / (mask.sum() + 1e-8)
    else:
        return epe.mean()


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss module.
    Computes loss between student and teacher model outputs.
    """
    def __init__(
        self,
        loss_type: str = 'mse',
        temperature: float = 1.0,
    ):
        """
        Args:
            loss_type: Type of distillation loss ('mse', 'cosine', 'smooth_l1', 'epe')
            temperature: Temperature for softening outputs (applied to flow magnitude)
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        
        self.loss_fns = {
            'mse': flow_mse_loss,
            'cosine': flow_cosine_loss,
            'smooth_l1': flow_smooth_l1_loss,
            'epe': flow_epe_loss,
        }
        
        if loss_type not in self.loss_fns:
            raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(self.loss_fns.keys())}")
    
    def forward(
        self,
        student_outputs: Dict,
        teacher_outputs: Dict,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_outputs: Student model outputs dict with 'flow' key
            teacher_outputs: Teacher model outputs dict with 'flow' key
            mask: Optional valid mask [B, 1, H, W]
        
        Returns:
            Scalar distillation loss
        """
        student_flow = student_outputs['flow']
        teacher_flow = teacher_outputs['flow']
        
        # Apply temperature scaling (softens the flow targets)
        if self.temperature != 1.0:
            teacher_flow = teacher_flow / self.temperature
            student_flow = student_flow / self.temperature
        
        loss = self.loss_fns[self.loss_type](student_flow, teacher_flow, mask)
        
        # Scale by temperature^2 when using temperature (standard distillation practice)
        if self.temperature != 1.0:
            loss = loss * (self.temperature ** 2)
        
        return loss


class DistillationCombinedLoss(nn.Module):
    """
    Combined loss for distillation training.
    Blends ground-truth supervision with teacher distillation.
    
    Loss = alpha * L_distill(student, teacher) + (1 - alpha) * L_gt(student, ground_truth)
    """
    def __init__(
        self,
        gt_criterion: nn.Module,
        distill_loss_type: str = 'mse',
        distill_temperature: float = 1.0,
        alpha: float = 0.5,
    ):
        """
        Args:
            gt_criterion: Ground-truth loss module (e.g., CombinedLoss)
            distill_loss_type: Type of distillation loss
            distill_temperature: Temperature for distillation
            alpha: Weight for distillation loss (0 = pure GT, 1 = pure distillation)
        """
        super().__init__()
        self.gt_criterion = gt_criterion
        self.distill_criterion = DistillationLoss(
            loss_type=distill_loss_type,
            temperature=distill_temperature,
        )
        self.alpha = alpha
    
    def forward(
        self,
        student_outputs: Dict,
        teacher_outputs: Dict,
        gt_flow: torch.Tensor,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation + ground-truth loss.
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs (can be None if alpha=0)
            gt_flow: Ground truth flow
            inputs: Input events (for GT loss computation)
            mask: Valid mask
        
        Returns:
            Dictionary of losses including 'total_loss'
        """
        losses = {}
        
        # Ground-truth loss
        gt_losses = self.gt_criterion(student_outputs, gt_flow, inputs, mask)
        losses['gt_total_loss'] = gt_losses['total_loss']
        losses['endpoint_loss'] = gt_losses['endpoint_loss']
        losses['angular_loss'] = gt_losses['angular_loss']
        losses['epe_ang_loss'] = gt_losses['epe_ang_loss']
        losses['dir_balance_loss'] = gt_losses.get('dir_balance_loss', torch.tensor(0.0))
        
        # Distillation loss
        if teacher_outputs is not None and self.alpha > 0:
            distill_loss = self.distill_criterion(student_outputs, teacher_outputs, mask)
            losses['distill_loss'] = distill_loss
        else:
            losses['distill_loss'] = torch.tensor(0.0, device=gt_flow.device)
        
        # Combined loss
        losses['total_loss'] = (
            self.alpha * losses['distill_loss'] + 
            (1 - self.alpha) * losses['gt_total_loss']
        )
        
        return losses
