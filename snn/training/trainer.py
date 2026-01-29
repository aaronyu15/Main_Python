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
from ..utils.metrics import (calculate_outliers, 
                             calculate_effective_epe, calculate_multi_percentile_epe)


class SNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
        logger: Optional[Logger] = None 
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger 
        
        self.criterion = CombinedLoss(
            endpoint_weight=config.get('endpoint_weight', 1.0),
            angular_weight=config.get('angular_weight', 0.5),
        )
        
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-4)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.get('lr_milestones', [100, 150, 200]),
            gamma=config.get('lr_gamma', 0.5)
        )
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_epe = float('inf')
        
        
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'endpoint_loss': 0.0,
            'angular_loss': 0.0,
        }
        epoch_outliers = 0.0
        epoch_flow_max = 0.0
        epoch_flow_min = 0.0
        epoch_flow_avg = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device)
            gt_flow = batch['flow'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            losses = self.criterion(outputs, gt_flow, valid_mask)
            
            losses['total_loss'].backward()

            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():

                outliers = calculate_outliers(outputs['flow'], gt_flow, valid_mask, threshold=3.0)
                
                epe_effective = calculate_effective_epe(outputs['flow'], gt_flow, valid_mask, threshold=0.1)
                
                percentile_metrics = calculate_multi_percentile_epe(
                    outputs['flow'], gt_flow, valid_mask, 
                    percentiles=[50, 75, 90, 95]
                )
                
                flow_pred = outputs['flow']
                flow_max = flow_pred.max().item()
                flow_min = flow_pred.min().item()
                flow_avg = flow_pred.abs().mean().item()
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            epoch_outliers += outliers            
            epoch_flow_max += flow_max
            epoch_flow_min += flow_min
            epoch_flow_avg += flow_avg
            num_batches += 1

            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'epe': losses['endpoint_loss'].item(),
                'outliers': f'{outliers:.2f}%',
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            if self.global_step % self.config.get('log_interval', 10) == 0:
                for key, value in losses.items():
                    self.logger.log_scalar(f'train/{key}', value, self.global_step)

                self.logger.log_scalar('train/outliers', outliers, self.global_step)
                self.logger.log_scalar('train/epe_effective', epe_effective, self.global_step)
                
                for key, value in percentile_metrics.items():
                    if 'epe' in key:
                        self.logger.log_scalar(f'train/{key}', value, self.global_step)
                
                self.logger.log_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                self.logger.log_scalar('train/flow_max', flow_max, self.global_step)
                self.logger.log_scalar('train/flow_min', flow_min, self.global_step)
                self.logger.log_scalar('train/flow_avg', flow_avg, self.global_step)
                
            
            self.global_step += 1
        
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        epoch_outliers /= num_batches
        epoch_flow_max /= num_batches
        epoch_flow_min /= num_batches
        epoch_flow_avg /= num_batches
        
        return {
            **epoch_losses,
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
            'endpoint_loss': 0.0,
            'angular_loss': 0.0
        }
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
            losses = self.criterion(outputs, gt_flow, valid_mask)
            
            # Compute metrics
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
            
            val_losses['total_loss'] += losses['total_loss']
            val_losses['endpoint_loss'] += losses['endpoint_loss']
            val_losses['angular_loss'] += losses['angular_loss']
            val_outliers += outliers

            val_epe_effective += epe_effective
            val_epe_top50pct += percentile_metrics['epe_top50pct']
            val_epe_top25pct += percentile_metrics['epe_top25pct']
            val_epe_top10pct += percentile_metrics['epe_top10pct']
            val_epe_top5pct += percentile_metrics['epe_top5pct']

            val_flow_max += flow_max
            val_flow_min += flow_min
            val_flow_avg += flow_avg

            num_batches += 1
        
        # Average
        for key in val_losses:
            val_losses[key] /= num_batches
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
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, EPE: {train_metrics['endpoint_loss']:.4f}, Ang: {train_metrics['angular_loss']:.4f}, Outliers: {train_metrics['outliers']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, EPE: {val_metrics['endpoint_loss']:.4f}, Ang: {val_metrics['angular_loss']:.4f}, Outliers: {val_metrics['outliers']:.2f}%")
            print(f"  Val EPE (Effective) - All: {val_metrics['endpoint_loss']:.4f}, Flow>0.1: {val_metrics['epe_effective']:.4f}")
            print(f"  Val EPE (Percentiles) - Top50%: {val_metrics['epe_top50pct']:.4f}, Top25%: {val_metrics['epe_top25pct']:.4f}, Top10%: {val_metrics['epe_top10pct']:.4f}, Top5%: {val_metrics['epe_top5pct']:.4f}")
            
            for key, value in train_metrics.items():
                self.logger.log_scalar(f'epoch/train_{key}', value, epoch)
            for key, value in val_metrics.items():
                self.logger.log_scalar(f'epoch/val_{key}', value, epoch)
            
            # Save checkpoint
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Save best model
            if val_metrics['endpoint_loss'] < self.best_val_epe:
                self.best_val_epe = val_metrics['endpoint_loss']
                self.save_checkpoint(is_best=True)
                print(f"  New best validation EPE: {self.best_val_epe:.4f}")
        
        print("Training completed!")
        print(f"Best validation EPE: {self.best_val_epe:.4f}")
