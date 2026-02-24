"""
Training script for knowledge distillation.

Usage:
    # Pure distillation (student learns only from teacher)
    python train_distill.py --config ./snn/configs/event_snn_lite_distill.yaml \
        --teacher-checkpoint ./checkpoints/teacher/best_model.pth \
        --checkpoint-dir ./checkpoints/student_distill/ \
        --log-dir ./logs/student_distill/

    # Hybrid training (student learns from both teacher and ground-truth)
    python train_distill.py --config ./snn/configs/event_snn_lite_distill.yaml \
        --teacher-checkpoint ./checkpoints/teacher/best_model.pth \
        --distill-alpha 0.5 \
        --checkpoint-dir ./checkpoints/student_hybrid/ \
        --log-dir ./logs/student_hybrid/
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from snn.models import EventSNNFlowNetLite
from snn.dataset import OpticalFlowDataset
from snn.training import DistillationTrainer
from snn.utils.logger import Logger
import random

from utils import load_config, build_model, load_teacher_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train SNN with Knowledge Distillation')
    
    # Config
    parser.add_argument('--config', type=str, required=True,
                      help='Path to student model configuration file')
    
    # Teacher model
    parser.add_argument('--teacher-checkpoint', type=str, required=True,
                      help='Path to pre-trained teacher model checkpoint')
    
    # Distillation settings (override config)
    parser.add_argument('--distill-alpha', type=float, default=None,
                      help='Weight for distillation loss (0=pure GT, 1=pure distillation). Overrides config.')
    parser.add_argument('--distill-temperature', type=float, default=None,
                      help='Temperature for softening outputs. Overrides config.')
    parser.add_argument('--distill-loss-type', type=str, default=None,
                      choices=['mse', 'cosine', 'smooth_l1', 'epe'],
                      help='Type of distillation loss. Overrides config.')
    
    # Training settings
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to student checkpoint to resume from')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/distill',
                      help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs/distill',
                      help='Directory for logs')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Data settings
    parser.add_argument('--train-data-root', type=str, default=None,
                      help='Override training dataset root')
    parser.add_argument('--val-data-root', type=str, default=None,
                      help='Validation dataset root')
    
    return parser.parse_args()


def build_dataloaders(config: dict, train_root: str = None, val_root: str = None):
    """Build training and validation data loaders."""
    train_root = train_root or config.get('data_root', '../blink_sim/output/train_set')
    val_root = val_root or config.get('val_data_root', '../blink_sim/output/valid_set')

    # Train dataset
    train_config = config.copy()
    train_config['data_root'] = train_root
    train_dataset = OpticalFlowDataset(config=train_config)

    # Val dataset
    val_config = config.copy()
    val_config['data_root'] = val_root
    val_dataset = OpticalFlowDataset(config=val_config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    args = parse_args()
    
    # Load student config
    config = load_config(args.config)
    print(f"Loaded student configuration from {args.config}")
    
    # Override distillation settings from command line
    if 'distillation' not in config:
        config['distillation'] = {}
    
    if args.distill_alpha is not None:
        config['distillation']['alpha'] = args.distill_alpha
    if args.distill_temperature is not None:
        config['distillation']['temperature'] = args.distill_temperature
    if args.distill_loss_type is not None:
        config['distillation']['loss_type'] = args.distill_loss_type
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Build student model
    student_model = build_model(config, device=device, train=True)
    print(f"Built student model: {config.get('model_type', 'EventSNNFlowNetLite')}")
    
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"Student parameters: {student_params:,}")
    
    # Load teacher model
    teacher_model = load_teacher_model(args.teacher_checkpoint, device=device)
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Compression ratio: {teacher_params / student_params:.2f}x")
    
    # Logger
    logger = Logger(log_dir=args.log_dir)
    student_model.set_logger(logger)
    
    # Print distillation settings
    distill_config = config.get('distillation', {})
    print(f"\nDistillation settings:")
    print(f"  Alpha: {distill_config.get('alpha', 0.5)} (0=pure GT, 1=pure teacher)")
    print(f"  Temperature: {distill_config.get('temperature', 1.0)}")
    print(f"  Loss type: {distill_config.get('loss_type', 'mse')}")
    
    # Build dataloaders
    train_root = args.train_data_root or config.get('data_root', None)
    val_root = args.val_data_root or config.get('val_data_root', None)
    train_loader, val_loader = build_dataloaders(config, train_root, val_root)

    print(f"\nData:")
    print(f"  Train data root: {train_root}")
    print(f"  Val data root: {val_root}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Create trainer
    trainer = DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        logger=logger 
    )
    
    # Log configuration
    logger.log_config(config, model=student_model)
    
    # Train
    num_epochs = config.get('num_epochs', 100)
    trainer.train(num_epochs=num_epochs, resume=args.resume)


if __name__ == '__main__':
    main()
