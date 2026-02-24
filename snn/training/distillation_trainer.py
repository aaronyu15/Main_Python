"""
Distillation trainer for knowledge transfer from teacher to student models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from pathlib import Path
from typing import Dict, Optional
import time
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..utils.logger import Logger
from ..utils.visualization import visualize_flow
from .losses import CombinedLoss, effective_epe, calculate_outliers, endpoint_error, angular_error
from .distillation_losses import DistillationCombinedLoss


class DistillationTrainer:
    """
    Trainer for knowledge distillation from a teacher model to a student model.
    
    Supports three training modes:
    - Pure distillation (alpha=1.0): Student learns only from teacher outputs
    - Pure ground-truth (alpha=0.0): Regular training without teacher
    - Hybrid (0 < alpha < 1): Student learns from both teacher and ground-truth
    """
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
        logger: Optional[Logger] = None,
    ):
        """
        Args:
            student_model: Student model to train
            teacher_model: Pre-trained teacher model (frozen)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logs
            logger: Optional logger instance
        """
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device) if teacher_model is not None else None
        self.config = config
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger
        
        # Freeze teacher model
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            print("Teacher model frozen for distillation")
        
        # Distillation settings
        distill_config = config.get('distillation', {})
        self.alpha = distill_config.get('alpha', 0.5)
        self.distill_loss_type = distill_config.get('loss_type', 'mse')
        self.distill_temperature = distill_config.get('temperature', 1.0)
        
        print(f"Distillation settings: alpha={self.alpha}, loss_type={self.distill_loss_type}, temperature={self.distill_temperature}")
        
        # Ground-truth criterion
        gt_criterion = CombinedLoss(
            endpoint_weight=config.get('endpoint_weight', 1.0),
            angular_weight=config.get('angular_weight', 0.5),
            epe_ang_weight=config.get('epe_ang_weight', 1.0),
            smoothness_weight=config.get('smoothness_weight', 1.0),
            vertical_weight=config.get('vertical_weight', 1.0),
            null_pred_weight=0,
            effective_epe_weights=config.get('effective_epe_weights', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            epe_type=config.get('epe_type', 'log'),
            directional_balance_weight=config.get('directional_balance_weight', 0.0),
        )
        
        # Combined distillation + GT loss
        self.criterion = DistillationCombinedLoss(
            gt_criterion=gt_criterion,
            distill_loss_type=self.distill_loss_type,
            distill_temperature=self.distill_temperature,
            alpha=self.alpha,
        )
        
        # Optimizer
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-4)
        self.optimizer = optim.AdamW(
            student_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.get('lr_milestones', [100, 150, 200]),
            gamma=config.get('lr_gamma', 0.5)
        )

        self.event_mask = True
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_epe = float('inf')
    
    def _visualize_batch(self, inputs: torch.Tensor, gt_flow: torch.Tensor, 
                         valid_mask: torch.Tensor, 
                         student_flow: Optional[torch.Tensor] = None,
                         teacher_flow: Optional[torch.Tensor] = None,
                         max_images: int = 4) -> Dict[str, torch.Tensor]:
        """
        Visualize batch data for TensorBoard logging.
        Includes both student and teacher predictions for comparison.
        """
        batch_size = min(inputs.shape[0], max_images)
        visualizations = {}
        
        # Visualize event input
        event_sum = inputs[:batch_size].sum(dim=(2), keepdim=True)
        event_vis = event_sum.repeat(1, 1, 3, 1, 1)
        visualizations['events'] = event_vis.view(-1, 3, event_vis.shape[3], event_vis.shape[4])
        
        # Visualize valid mask
        valid_vis = valid_mask[:batch_size].repeat(1, 1, 3, 1, 1)
        visualizations['valid_mask'] = valid_vis.view(-1, 3, valid_vis.shape[3], valid_vis.shape[4])
        
        max_flow = min(torch.norm(gt_flow, dim=1).max().item(), 1.0)
        
        # Visualize GT flow
        gt_flow_vis = []
        gt_flow_mask_vis = []

        for i in range(batch_size):
            flow_np = gt_flow[i].cpu()  # [2, H, W]

            flow_mask = gt_flow[i] * valid_mask[i]  # Mask out invalid pixels
            flow_np_mask = flow_mask.cpu()  # [2, H, W]

            flow_color = visualize_flow(flow_np, max_flow=max_flow)
            flow_color_mask = visualize_flow(flow_np_mask, max_flow=max_flow)

            flow_color = torch.from_numpy(flow_color).permute(2, 0, 1).float() / 255.0
            flow_color_mask = torch.from_numpy(flow_color_mask).permute(2, 0, 1).float() / 255.0

            gt_flow_vis.append(flow_color)
            gt_flow_mask_vis.append(flow_color_mask)

        visualizations['gt_flow'] = torch.stack(gt_flow_vis)  # [B, 3, H, W]
        visualizations['gt_flow_masked'] = torch.stack(gt_flow_mask_vis)  # [B, 3, H, W]
        
        # Visualize student flow
        if student_flow is not None:
            student_flow_vis = []
            student_flow_mask_vis = []
            for i in range(batch_size):
                flow_np = student_flow[i].cpu()

                flow_mask = student_flow[i] * valid_mask[i]  # Mask out invalid pixels
                flow_np_mask = flow_mask.cpu()

                flow_color = visualize_flow(flow_np, max_flow=max_flow)
                flow_color_mask = visualize_flow(flow_np_mask, max_flow=max_flow)

                flow_color = torch.from_numpy(flow_color).permute(2, 0, 1).float() / 255.0
                flow_color_mask = torch.from_numpy(flow_color_mask).permute(2, 0, 1).float() / 255.0

                student_flow_vis.append(flow_color)
                student_flow_mask_vis.append(flow_color_mask)
            visualizations['student_flow'] = torch.stack(student_flow_vis)
            visualizations['student_flow_masked'] = torch.stack(student_flow_mask_vis)
        
        # Visualize teacher flow
        if teacher_flow is not None:
            teacher_flow_vis = []
            teacher_flow_mask_vis = []
            for i in range(batch_size):
                flow_np = teacher_flow[i].cpu()
                flow_mask = teacher_flow[i] * valid_mask[i]  # Mask out invalid pixels
                flow_np_mask = flow_mask.cpu()
                flow_color = visualize_flow(flow_np, max_flow=max_flow)
                flow_color_mask = visualize_flow(flow_np_mask, max_flow=max_flow)
                flow_color = torch.from_numpy(flow_color).permute(2, 0, 1).float() / 255.0
                flow_color_mask = torch.from_numpy(flow_color_mask).permute(2, 0, 1).float() / 255.0
                teacher_flow_vis.append(flow_color)
                teacher_flow_mask_vis.append(flow_color_mask)
            visualizations['teacher_flow'] = torch.stack(teacher_flow_vis)
            visualizations['teacher_flow_masked'] = torch.stack(teacher_flow_mask_vis)
        
        return visualizations
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.student_model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'gt_total_loss': 0.0,
            'distill_loss': 0.0,
            'endpoint_loss': 0.0,
            'angular_loss': 0.0,
            'epe_ang_loss': 0.0,
            'dir_balance_loss': 0.0,
        }
        epoch_outliers = 0.0
        epoch_flow_max = 0.0
        epoch_flow_avg = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device)
            gt_flow = batch['flow'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Student forward pass
            student_outputs = self.student_model(inputs)
            
            # Teacher forward pass (no gradients)
            teacher_outputs = None
            if self.teacher_model is not None and self.alpha > 0:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(inputs)

            # Event mask
            if self.event_mask:
                activity_patch = inputs.sum(dim=(1, 2))
                low_activity_mask = (activity_patch < 1)
                low_activity_mask = low_activity_mask.unsqueeze(1)
                valid_mask[low_activity_mask] = 0.0
            
            # Compute losses
            losses = self.criterion(student_outputs, teacher_outputs, gt_flow, inputs, valid_mask)
            
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                outliers = calculate_outliers(student_outputs['flow'], gt_flow, valid_mask, threshold=1.0)
                
                flow_pred = student_outputs['flow']
                flow_mag = torch.norm(flow_pred, dim=1)
                flow_max = flow_mag.max().item()
                flow_avg = flow_mag.abs().mean().item()
            
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += float(losses[key])
            epoch_outliers += outliers            
            epoch_flow_max += flow_max
            epoch_flow_avg += flow_avg
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'distill': losses['distill_loss'].item() if torch.is_tensor(losses['distill_loss']) else losses['distill_loss'],
                'gt': losses['gt_total_loss'].item(),
                'outliers': f'{outliers:.2f}%',
            })
            
            # Log scalars
            if self.global_step % self.config.get('log_interval', 10) == 0:
                for key, value in losses.items():
                    val = value.item() if torch.is_tensor(value) else value
                    self.logger.log_scalar(f'train/{key}', val, self.global_step)

                self.logger.log_scalar('train/outliers', outliers, self.global_step)
                self.logger.log_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                self.logger.log_scalar('train/flow_max', flow_max, self.global_step)
                self.logger.log_scalar('train/flow_avg', flow_avg, self.global_step)
            
            # Log images
            if self.global_step % self.config.get('image_log_interval', 100) == 0:
                teacher_flow = teacher_outputs['flow'] if teacher_outputs else None
                visualizations = self._visualize_batch(
                    inputs, gt_flow, valid_mask, 
                    student_flow=student_outputs['flow'],
                    teacher_flow=teacher_flow,
                    max_images=self.config.get('max_images_log', 4)
                )
                
                for vis_name, vis_tensor in visualizations.items():
                    if vis_name == 'events' or vis_name == 'valid_mask':
                        nrow = self.config.get('num_bins', 10)
                    else:
                        nrow = 2
                    grid = make_grid(vis_tensor, nrow=nrow, normalize=False, pad_value=1.0)
                    self.logger.log_image(f'train/{vis_name}', grid, self.global_step)
            
            self.global_step += 1
        
        # Average
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        epoch_outliers /= num_batches
        epoch_flow_max /= num_batches
        epoch_flow_avg /= num_batches
        
        return {
            **epoch_losses,
            'outliers': epoch_outliers,
            'flow_max': epoch_flow_max,
            'flow_avg': epoch_flow_avg,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.student_model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'gt_total_loss': 0.0,
            'distill_loss': 0.0,
            'endpoint_loss': 0.0,
            'angular_loss': 0.0,
            'epe_ang_loss': 0.0,
            'dir_balance_loss': 0.0,
        }
        val_epe_effective = 0.0
        val_outliers = 0.0
        val_flow_max = 0.0
        val_flow_avg = 0.0
        num_batches = 0
        
        first_batch_vis = None
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
            inputs = batch['input'].to(self.device)
            gt_flow = batch['flow'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)

            if self.event_mask:
                activity_patch = inputs.sum(dim=(1,2))
                low_activity_mask = (activity_patch < 1)
                low_activity_mask = low_activity_mask.unsqueeze(1)
                valid_mask[low_activity_mask] = 0.0
            
            # Student forward
            student_outputs = self.student_model(inputs)
            
            # Teacher forward
            teacher_outputs = None
            if self.teacher_model is not None and self.alpha > 0:
                teacher_outputs = self.teacher_model(inputs)
            
            # Compute losses
            losses = self.criterion(student_outputs, teacher_outputs, gt_flow, inputs, valid_mask)
            
            # Metrics
            outliers = calculate_outliers(student_outputs['flow'], gt_flow, valid_mask, threshold=1.0)
            epe_effective = effective_epe(student_outputs['flow'], gt_flow, valid_mask, threshold_min=0.1)
            
            flow_pred = student_outputs['flow']
            flow_mag = torch.norm(flow_pred, dim=1)
            flow_max = flow_mag.max().item()
            flow_avg = flow_mag.abs().mean().item()
            
            for key in val_losses:
                if key in losses:
                    val_losses[key] += float(losses[key])
            val_outliers += outliers
            val_epe_effective += epe_effective
            val_flow_max += flow_max
            val_flow_avg += flow_avg
            num_batches += 1
            
            # Store first batch for visualization
            if batch_idx == 0:
                first_batch_vis = {
                    'inputs': inputs,
                    'gt_flow': gt_flow,
                    'valid_mask': valid_mask,
                    'student_flow': student_outputs['flow'],
                    'teacher_flow': teacher_outputs['flow'] if teacher_outputs else None,
                }
        
        # Average
        for key in val_losses:
            val_losses[key] /= num_batches
        val_epe_effective /= num_batches
        val_outliers /= num_batches
        val_flow_max /= num_batches
        val_flow_avg /= num_batches
        
        # Log validation images
        if first_batch_vis is not None:
            visualizations = self._visualize_batch(
                first_batch_vis['inputs'],
                first_batch_vis['gt_flow'],
                first_batch_vis['valid_mask'],
                student_flow=first_batch_vis['student_flow'],
                teacher_flow=first_batch_vis['teacher_flow'],
                max_images=self.config.get('max_images_log', 4)
            )
            
            for vis_name, vis_tensor in visualizations.items():
                if vis_name == 'events' or vis_name == 'valid_mask':
                    nrow = self.config.get('num_bins', 10)
                else:
                    nrow = 2
                grid = make_grid(vis_tensor, nrow=nrow, normalize=False, pad_value=1.0)
                self.logger.log_image(f'val/{vis_name}', grid, self.global_step)
        
        return {
            **val_losses,
            'epe_effective': val_epe_effective,
            'outliers': val_outliers,
            'flow_max': val_flow_max,
            'flow_avg': val_flow_avg,
        }
    
    def save_checkpoint(self, filename: str = 'checkpoint.pth', is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_epe': self.best_val_epe,
            'config': self.config,
            'distillation': {
                'alpha': self.alpha,
                'loss_type': self.distill_loss_type,
                'temperature': self.distill_temperature,
            }
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")
        
        if is_best:
            best_filepath = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_filepath)
            print(f"Saved best model to {best_filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.student_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_epe = checkpoint['best_val_epe']
        
        print(f"Loaded checkpoint from {filepath} (epoch {self.epoch})")

    def update_settings(self, epoch: int):
        """Update training settings based on epoch."""
        if self.config.get('event_mask_disable_epoch') is not None:
            if epoch >= self.config['event_mask_disable_epoch'] and self.event_mask:
                self.event_mask = False
                print(f"Epoch {epoch}: Disabled event mask for training")

        if self.config.get('disable_skip_epoch') is not None:
            if epoch >= self.config['disable_skip_epoch'] and self.student_model.disable_skip is False:
                self.student_model.disable_skip = True
                print(f"Epoch {epoch}: Disabled skip connections for training")

    def train(self, num_epochs: int, resume: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume: Path to checkpoint to resume from
        """
        if resume is not None:
            self.load_checkpoint(resume)
        
        print(f"Starting distillation training from epoch {self.epoch}")
        print(f"Distillation alpha: {self.alpha}")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        start_epoch = self.epoch + 1

        for epoch in range(start_epoch, num_epochs+1):
            self.epoch = epoch

            self.update_settings(epoch)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, Distill: {train_metrics['distill_loss']:.4f}, GT: {train_metrics['gt_total_loss']:.4f}, Outliers: {train_metrics['outliers']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, Distill: {val_metrics['distill_loss']:.4f}, GT: {val_metrics['gt_total_loss']:.4f}, Outliers: {val_metrics['outliers']:.2f}%")
            print(f"  Val EPE (Effective) - All: {val_metrics['endpoint_loss']:.4f}, Flow>0.1: {val_metrics['epe_effective']:.4f}")
            
            for key, value in train_metrics.items():
                self.logger.log_scalar(f'epoch/train_{key}', value, epoch)
            for key, value in val_metrics.items():
                self.logger.log_scalar(f'epoch/val_{key}', value, epoch)
            
            # Save checkpoint
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Save best model (based on GT epe_ang_loss for fair comparison)
            if val_metrics['epe_ang_loss'] < self.best_val_epe:
                self.best_val_epe = val_metrics['epe_ang_loss']
                self.save_checkpoint(is_best=True)
                print(f"  New best validation EPE: {self.best_val_epe:.4f}")
        
        print("Distillation training completed!")
        print(f"Best validation EPE: {self.best_val_epe:.4f}")
