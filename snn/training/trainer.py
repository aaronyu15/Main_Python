"""
SNN Trainer Class
Main training loop with quantization awareness and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import time
import json
from tqdm import tqdm

from .losses import CombinedLoss, endpoint_error
from ..utils.logger import Logger
from ..utils.metrics import (compute_metrics, calculate_outliers, 
                             calculate_effective_epe, calculate_multi_percentile_epe)


class SNNTrainer:
    """
    Trainer for Spiking Neural Networks on Optical Flow
    
    Features:
    - Quantization-aware training with switches
    - Progressive quantization (full -> 8-bit -> 4-bit -> binary)
    - Checkpointing and resume
    - Tensorboard logging
    - Learning rate scheduling
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.logger = Logger(log_dir)
        
        # Loss function
        self.criterion = CombinedLoss(
            flow_weight=config.get('flow_weight', 1.0),
            smooth_weight=config.get('smooth_weight', 0.1),
            sparsity_weight=config.get('sparsity_weight', 0.01),
            quant_weight=config.get('quant_weight', 0.0001),
            target_spike_rate=config.get('target_spike_rate', 0.1)
        )
        
        # Optimizer
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-4)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.get('lr_milestones', [100, 150, 200]),
            gamma=config.get('lr_gamma', 0.5)
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_epe = float('inf')
        
        # Quantization schedule
        self.quantization_enabled = config.get('quantization_enabled', False)
        
        # If no schedule provided, use initial_bit_width for all epochs
        if 'quantization_schedule' in config:
            self.quantization_schedule = config['quantization_schedule']
        else:
            # Create simple schedule: keep initial bit-width throughout training
            initial_bw = config.get('initial_bit_width', 8)
            self.quantization_schedule = {0: initial_bw}
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'flow_loss': 0.0,
            'smooth_loss': 0.0,
            'sparsity_loss': 0.0,
            'quant_loss': 0.0
        }
        epoch_epe = 0.0
        epoch_outliers = 0.0
        epoch_flow_max = 0.0
        epoch_flow_min = 0.0
        epoch_flow_avg = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            inputs = batch['input'].to(self.device)
            gt_flow = batch['flow'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute losses
            losses = self.criterion(outputs, gt_flow, valid_mask, self.model)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                epe = endpoint_error(outputs['flow'], gt_flow, valid_mask)
                outliers = calculate_outliers(outputs['flow'], gt_flow, valid_mask, threshold=3.0)
                
                # Effective pixel metrics (flow magnitude > 0.1)
                epe_effective = calculate_effective_epe(outputs['flow'], gt_flow, valid_mask, threshold=0.1)
                
                # Percentile-based metrics (top 50%, 25%, 10%, 5%)
                percentile_metrics = calculate_multi_percentile_epe(
                    outputs['flow'], gt_flow, valid_mask, 
                    percentiles=[50, 75, 90, 95]
                )
                
                flow_pred = outputs['flow']
                flow_max = flow_pred.max().item()
                flow_min = flow_pred.min().item()
                flow_avg = flow_pred.abs().mean().item()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            epoch_epe += epe.item()            
            epoch_outliers += outliers            
            epoch_flow_max += flow_max
            epoch_flow_min += flow_min
            epoch_flow_avg += flow_avg
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'epe': epe.item(),
                'outliers': f'{outliers:.2f}%',
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to tensorboard
            if self.global_step % self.config.get('log_interval', 10) == 0:
                for key, value in losses.items():
                    self.logger.log_scalar(f'train/{key}', value, self.global_step)
                self.logger.log_scalar('train/epe', epe.item(), self.global_step)
                self.logger.log_scalar('train/epe_effective', epe_effective, self.global_step)
                
                # Log percentile-based EPE
                for key, value in percentile_metrics.items():
                    if 'epe' in key:
                        self.logger.log_scalar(f'train/{key}', value, self.global_step)
                
                self.logger.log_scalar('train/outliers', outliers, self.global_step)
                self.logger.log_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                # Log flow prediction statistics
                self.logger.log_scalar('train/flow_max', flow_max, self.global_step)
                self.logger.log_scalar('train/flow_min', flow_min, self.global_step)
                self.logger.log_scalar('train/flow_avg', flow_avg, self.global_step)
                
                # Log spike statistics
                if 'spike_stats' in outputs:
                    for key, value in outputs['spike_stats'].items():
                        if isinstance(value, (int, float)):
                            self.logger.log_scalar(f'train/spike_{key}', value, self.global_step)
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        epoch_epe /= num_batches
        epoch_outliers /= num_batches
        epoch_flow_max /= num_batches
        epoch_flow_min /= num_batches
        epoch_flow_avg /= num_batches
        
        return {
            **epoch_losses,
            'epe': epoch_epe,
            'outliers': epoch_outliers,
            'flow_max': epoch_flow_max,
            'flow_min': epoch_flow_min,
            'flow_avg': epoch_flow_avg
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'flow_loss': 0.0
        }
        val_epe = 0.0
        val_epe_effective = 0.0
        val_epe_top50pct = 0.0
        val_epe_top25pct = 0.0
        val_epe_top10pct = 0.0
        val_epe_top5pct = 0.0
        val_outliers = 0.0
        val_flow_max = 0.0
        val_flow_min = 0.0
        val_flow_avg = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move to device
            inputs = batch['input'].to(self.device)
            gt_flow = batch['flow'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute losses
            losses = self.criterion(outputs, gt_flow, valid_mask, self.model)
            
            # Compute metrics
            epe = endpoint_error(outputs['flow'], gt_flow, valid_mask)
            outliers = calculate_outliers(outputs['flow'], gt_flow, valid_mask, threshold=3.0)
            
            # Effective pixel metrics
            epe_effective = calculate_effective_epe(outputs['flow'], gt_flow, valid_mask, threshold=0.1)
            
            # Percentile-based metrics
            percentile_metrics = calculate_multi_percentile_epe(
                outputs['flow'], gt_flow, valid_mask,
                percentiles=[50, 75, 90, 95]
            )
            
            flow_pred = outputs['flow']
            flow_max = flow_pred.max().item()
            flow_min = flow_pred.min().item()
            flow_avg = flow_pred.abs().mean().item()
            
            # Accumulate
            val_losses['total_loss'] += losses['total_loss'].item()
            val_losses['flow_loss'] += losses['flow_loss'].item()
            val_epe += epe.item()
            val_epe_effective += epe_effective
            val_epe_top50pct += percentile_metrics['epe_top50pct']
            val_epe_top25pct += percentile_metrics['epe_top25pct']
            val_epe_top10pct += percentile_metrics['epe_top10pct']
            val_epe_top5pct += percentile_metrics['epe_top5pct']
            val_outliers += outliers
            val_flow_max += flow_max
            val_flow_min += flow_min
            val_flow_avg += flow_avg
            num_batches += 1
        
        # Average
        for key in val_losses:
            val_losses[key] /= num_batches
        val_epe /= num_batches
        val_epe_effective /= num_batches
        val_epe_top50pct /= num_batches
        val_epe_top25pct /= num_batches
        val_epe_top10pct /= num_batches
        val_epe_top5pct /= num_batches
        val_outliers /= num_batches
        val_flow_max /= num_batches
        val_flow_min /= num_batches
        val_flow_avg /= num_batches
        
        return {
            **val_losses,
            'epe': val_epe,
            'epe_effective': val_epe_effective,
            'epe_top50pct': val_epe_top50pct,
            'epe_top25pct': val_epe_top25pct,
            'epe_top10pct': val_epe_top10pct,
            'epe_top5pct': val_epe_top5pct,
            'outliers': val_outliers,
            'flow_max': val_flow_max,
            'flow_min': val_flow_min,
            'flow_avg': val_flow_avg
        }
    
    def update_quantization(self):
        """Update quantization bit-width based on schedule"""
        if not self.quantization_enabled:
            return
        
        # Find current bit width from schedule
        current_bit_width = None
        for epoch_threshold in sorted(self.quantization_schedule.keys(), reverse=True):
            if self.epoch >= epoch_threshold:
                current_bit_width = self.quantization_schedule[epoch_threshold]
                break
        
        if current_bit_width is None:
            return  # No schedule entry applies yet
        
        # Update model quantization bit-width
        updated = False
        for module in self.model.modules():
            if hasattr(module, 'bit_width'):
                if module.bit_width != current_bit_width:
                    if not updated:  # Only print once
                        print(f"Epoch {self.epoch}: Updating quantization {module.bit_width}-bit -> {current_bit_width}-bit")
                        updated = True
                    module.bit_width = current_bit_width
                    
                    # Update quantization layer if it exists
                    if hasattr(module, 'quant_layer') and module.quant_layer is not None:
                        module.quant_layer.bit_width = current_bit_width
                        # Reset running statistics for new bit-width
                        module.quant_layer.num_batches_tracked.zero_()
                        
                        # Update qmin/qmax for new bit-width
                        if module.quant_layer.symmetric:
                            module.quant_layer.qmin = -(2 ** (current_bit_width - 1))
                            module.quant_layer.qmax = 2 ** (current_bit_width - 1) - 1
                        else:
                            module.quant_layer.qmin = 0
                            module.quant_layer.qmax = 2 ** current_bit_width - 1
    
    def save_checkpoint(self, filename: str = 'checkpoint.pth', is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_epe': self.best_val_epe,
            'config': self.config
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")
        
        if is_best:
            best_filepath = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_filepath)
            print(f"Saved best model to {best_filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_epe = checkpoint['best_val_epe']
        
        print(f"Loaded checkpoint from {filepath} (epoch {self.epoch})")
    
    def train(self, num_epochs: int, resume: Optional[str] = None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            resume: Path to checkpoint to resume from
        """
        # Resume if checkpoint provided
        if resume is not None:
            self.load_checkpoint(resume)
        
        print(f"Starting training from epoch {self.epoch}")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            
            # Update quantization schedule
            self.update_quantization()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, EPE: {train_metrics['epe']:.4f}, Outliers: {train_metrics['outliers']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, EPE: {val_metrics['epe']:.4f}, Outliers: {val_metrics['outliers']:.2f}%")
            print(f"  Val EPE (Effective) - All: {val_metrics['epe']:.4f}, Flow>0.1: {val_metrics['epe_effective']:.4f}")
            print(f"  Val EPE (Percentiles) - Top50%: {val_metrics['epe_top50pct']:.4f}, Top25%: {val_metrics['epe_top25pct']:.4f}, Top10%: {val_metrics['epe_top10pct']:.4f}, Top5%: {val_metrics['epe_top5pct']:.4f}")
            
            for key, value in train_metrics.items():
                self.logger.log_scalar(f'epoch/train_{key}', value, epoch)
            for key, value in val_metrics.items():
                self.logger.log_scalar(f'epoch/val_{key}', value, epoch)
            
            # Save checkpoint
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Save best model
            if val_metrics['epe'] < self.best_val_epe:
                self.best_val_epe = val_metrics['epe']
                self.save_checkpoint(is_best=True)
                print(f"  New best validation EPE: {self.best_val_epe:.4f}")
        
        print("Training completed!")
        print(f"Best validation EPE: {self.best_val_epe:.4f}")
