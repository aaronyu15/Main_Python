"""
Quantization Script — supports both PTQ and QAT.

Modes:
  PTQ (Post-Training Quantization):
    - Load pretrained model, run calibration pass, export quantized model
    - No training required — just calibration over a few batches of data

  QAT (Quantization-Aware Training):
    - Load pretrained model, fine-tune with fake quantization (STE)
    - Weight scales derived from weight statistics (not learned)
    - Activation scales tracked via EMA of running min/max

Usage:
    # PTQ: calibrate and save quantized model (no training)
    python finetune_quantized.py \
        --config snn/configs/event_snn_lite_8bit.yaml \
        --pretrained checkpoints/student_main/best_model.pth \
        --mode ptq \
        --checkpoint-dir checkpoints/ptq_8bit
        --log-dir logs/ptq_8bit

    # QAT: fine-tune with 8-bit quantization
    python finetune_quantized.py \
        --config snn/configs/event_snn_lite_8bit.yaml \
        --pretrained checkpoints/teacher_10000u/best_model.pth \
        --mode qat \
        --checkpoint-dir checkpoints/qat_8bit \
        --log-dir logs/qat_8bit
"""
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from snn.models import EventSNNFlowNetLite
from snn.dataset import OpticalFlowDataset
from snn.training import SNNTrainer
from snn.utils.logger import Logger
from snn.models.quant_utils import (
    enable_quantization_warnings, reset_quantization_warning_counts,
    calibrate_model, set_quant_mode, print_scale_summary, export_quantized_params
)
from snn.utils.visualization import visualize_flow
from torchvision.utils import make_grid
import random
import warnings

from utils import load_config, get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Quantization-Aware Fine-tuning for SNN')
    
    # Config
    parser.add_argument('--config', type=str, required=True,
                      help='Path to quantized model configuration file')
    
    # Pretrained model
    parser.add_argument('--pretrained', type=str, required=True,
                      help='Path to pre-trained model checkpoint to fine-tune from')
    parser.add_argument('--mode', type=str, default='qat', choices=['ptq', 'qat'],
                      help='Quantization mode: ptq (calibrate only) or qat (fine-tune)')
    
    # Training settings
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume fine-tuning from')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/quantized',
                      help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs/quantized',
                      help='Directory for logs')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Data settings
    parser.add_argument('--train-data-root', type=str, default=None,
                      help='Override training dataset root')
    parser.add_argument('--val-data-root', type=str, default=None,
                      help='Validation dataset root')
    
    # Quantization warnings
    parser.add_argument('--disable-quant-warnings', action='store_true',
                      help='Disable quantization range warnings')
    parser.add_argument('--strict-load', action='store_true',
                      help='Use strict mode when loading pretrained weights')
    
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


def load_pretrained_weights(model: nn.Module, checkpoint_path: str, strict: bool = False) -> dict:
    """
    Load pretrained weights into the model.
    
    Handles weight loading from full-precision or different bit-width models.
    Uses non-strict loading by default to handle quantization-related parameter differences.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to pretrained checkpoint
        strict: Whether to require exact match of state dict keys
        
    Returns:
        Checkpoint dict containing metadata
    """
    print(f"Loading pretrained weights from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume checkpoint is the state dict itself
        state_dict = checkpoint
        checkpoint = {'state_dict': state_dict}
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"  Pretrained model trained for {checkpoint['epoch']} epochs")
    if 'best_val_epe' in checkpoint:
        print(f"  Pretrained best validation EPE: {checkpoint['best_val_epe']:.4f}")
    if 'config' in checkpoint:
        pretrained_config = checkpoint['config']
        print(f"  Pretrained quantization: W{pretrained_config.get('weight_bit_width', 32)}A{pretrained_config.get('act_bit_width', 32)}M{pretrained_config.get('mem_bit_width', 32)}")
    
    return checkpoint


def print_quantization_info(config: dict):
    """Print quantization settings."""
    print("\n" + "="*60)
    print("Quantization-Aware Fine-tuning Configuration")
    print("="*60)
    
    quant_weights = config.get('quantize_weights', False)
    quant_act = config.get('quantize_activations', False)
    quant_mem = config.get('quantize_mem', False)
    
    weight_bits = config.get('weight_bit_width', 8)
    act_bits = config.get('act_bit_width', 8)
    mem_bits = config.get('mem_bit_width', 16)
    
    print(f"Weight quantization:     {'Enabled' if quant_weights else 'Disabled'} ({weight_bits}-bit)")
    print(f"Activation quantization: {'Enabled' if quant_act else 'Disabled'} ({act_bits}-bit)")
    print(f"Membrane quantization:   {'Enabled' if quant_mem else 'Disabled'} ({mem_bits}-bit)")
    
    if quant_weights or quant_act or quant_mem:
        # Calculate theoretical compression
        if quant_weights:
            compression = 32.0 / weight_bits
            print(f"\nTheoretical weight compression: {compression:.1f}x")
        
        # Print quantization ranges
        print(f"\nQuantization ranges:")
        if quant_weights:
            qmax_w = 2 ** (weight_bits - 1) - 1
            print(f"  Weights ({weight_bits}-bit symmetric): [{-qmax_w-1}, {qmax_w}]")
        if quant_act:
            qmax_a = 2 ** act_bits - 1
            print(f"  Activations ({act_bits}-bit asymmetric): [0, {qmax_a}]")
        if quant_mem:
            qmax_m = 2 ** (mem_bits - 1) - 1
            print(f"  Membrane ({mem_bits}-bit symmetric): [{-qmax_m-1}, {qmax_m}]")
    
    print("="*60 + "\n")


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Setup quantization warnings
    if args.disable_quant_warnings:
        enable_quantization_warnings(False)
        print("Quantization range warnings disabled")
    else:
        enable_quantization_warnings(True)
        reset_quantization_warning_counts()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed: {args.seed}")
    
    # Print quantization info
    print_quantization_info(config)
    
    # Build model with quantization config
    model = get_model(config)
    print(f"Built model: {config.get('model_type', 'EventSNNFlowNetLite')}")
    
    # Load pretrained weights
    pretrained_checkpoint = load_pretrained_weights(
        model, 
        args.pretrained, 
        strict=args.strict_load
    )
    
    # Move to device
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Logger
    logger = Logger(log_dir=args.log_dir)
    model.set_logger(logger)
    
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
    trainer = SNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        logger=logger 
    )
    
    # Log configuration
    logger.log_config(config, model=model)
    
    # Add quantization info to logged config
    logger.log_text('quantization/pretrained_from', args.pretrained)
    logger.log_text('quantization/weight_bits', str(config.get('weight_bit_width', 8)))
    logger.log_text('quantization/act_bits', str(config.get('act_bit_width', 8)))
    logger.log_text('quantization/mem_bits', str(config.get('mem_bit_width', 16)))
    logger.log_text('quantization/mode', args.mode)
    
    if args.mode == 'ptq':
        # ---- PTQ: calibrate and save ----
        print(f"\n[PTQ] Running post-training quantization...")
        set_quant_mode(model, 'ptq')
        
        calibrate_model(
            model, train_loader,
            device=device,
            num_batches=config.get('num_calib_batches', 100),
            config=config,
        )
        
        # Save calibrated model
        save_path = Path(args.checkpoint_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'mode': 'ptq',
            'pretrained_from': args.pretrained,
        }, save_path / 'ptq_model.pth')
        
        print(f"\n[PTQ] Calibrated model saved to: {save_path / 'ptq_model.pth'}")
        
        # ---- PTQ: run validation with TensorBoard image logging ----
        print(f"\n[PTQ] Running evaluation on validation set...")
        model.eval()
        
        val_epe_total = 0.0
        num_val_batches = 0
        max_vis_images = config.get('max_images_log', 4)
        num_bins = config.get('num_bins', 5)
        vis_interval = config.get('log_interval', 10)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs = batch['input'].to(device)
                gt_flow = batch['flow'].to(device)
                valid_mask = batch['valid_mask'].to(device)
                
                outputs = model(inputs)
                pred_flow = outputs.get('flow', outputs.get('pred_flow',
                    list(outputs.values())[0])) if isinstance(outputs, dict) else outputs
                
                # Compute EPE
                epe = torch.norm(pred_flow - gt_flow, p=2, dim=1, keepdim=True)
                valid_epe = (epe * valid_mask).sum() / valid_mask.sum().clamp(min=1)
                val_epe_total += valid_epe.item()
                num_val_batches += 1
                
                # Log images every vis_interval batches
                if batch_idx % vis_interval == 0:
                    bs = min(inputs.shape[0], max_vis_images)
                    
                    # Events
                    event_sum = inputs[:bs].sum(dim=2, keepdim=True)
                    event_vis = event_sum.repeat(1, 1, 3, 1, 1)
                    grid = make_grid(event_vis.view(-1, 3, event_vis.shape[3], event_vis.shape[4]),
                                     nrow=num_bins, normalize=False, pad_value=1.0)
                    logger.log_image('ptq/events', grid, batch_idx)
                    
                    # Valid mask
                    valid_vis = valid_mask[:bs].repeat(1, 1, 3, 1, 1)
                    grid = make_grid(valid_vis.view(-1, 3, valid_vis.shape[3], valid_vis.shape[4]),
                                     nrow=num_bins, normalize=False, pad_value=1.0)
                    logger.log_image('ptq/valid_mask', grid, batch_idx)
                    
                    max_flow = min(torch.norm(gt_flow, dim=1).max().item(), 1.0)
                    
                    # GT flow, predicted flow, and masked versions
                    gt_vis, pred_vis = [], []
                    gt_mask_vis, pred_mask_vis = [], []
                    for i in range(bs):
                        gt_c = visualize_flow(gt_flow[i].cpu(), max_flow=max_flow)
                        gt_vis.append(torch.from_numpy(gt_c).permute(2,0,1).float() / 255.0)
                        
                        pr_c = visualize_flow(pred_flow[i].cpu(), max_flow=max_flow)
                        pred_vis.append(torch.from_numpy(pr_c).permute(2,0,1).float() / 255.0)
                        
                        gt_m = visualize_flow((gt_flow[i] * valid_mask[i]).cpu(), max_flow=max_flow)
                        gt_mask_vis.append(torch.from_numpy(gt_m).permute(2,0,1).float() / 255.0)
                        
                        pr_m = visualize_flow((pred_flow[i] * valid_mask[i]).cpu(), max_flow=max_flow)
                        pred_mask_vis.append(torch.from_numpy(pr_m).permute(2,0,1).float() / 255.0)
                    
                    logger.log_image('ptq/gt_flow',
                        make_grid(torch.stack(gt_vis), nrow=2, pad_value=1.0), batch_idx)
                    logger.log_image('ptq/pred_flow',
                        make_grid(torch.stack(pred_vis), nrow=2, pad_value=1.0), batch_idx)
                    logger.log_image('ptq/gt_flow_masked',
                        make_grid(torch.stack(gt_mask_vis), nrow=2, pad_value=1.0), batch_idx)
                    logger.log_image('ptq/pred_flow_masked',
                        make_grid(torch.stack(pred_mask_vis), nrow=2, pad_value=1.0), batch_idx)
                    
                    # Log per-batch EPE as a scalar too
                    logger.log_scalar('ptq/batch_epe', valid_epe.item(), batch_idx)
        
        val_epe_avg = val_epe_total / max(num_val_batches, 1)
        logger.log_scalar('ptq/val_epe', val_epe_avg, 0)
        
        print(f"[PTQ] Validation EPE: {val_epe_avg:.4f} (over {num_val_batches} batches)")
        print(f"[PTQ] TensorBoard images logged to: {args.log_dir}")
        
    else:
        # ---- QAT: fine-tune with fake quantization ----
        set_quant_mode(model, 'qat')
        
        num_epochs = config.get('num_epochs', 100)
        print(f"\n[QAT] Starting quantization-aware fine-tuning for {num_epochs} epochs...")
        trainer.train(num_epochs=num_epochs, resume=args.resume)
        
        print_scale_summary(model)
        
        print("\n" + "="*60)
        print("Quantization-Aware Fine-tuning Complete!")
        print("="*60)
        print(f"Best model saved to: {args.checkpoint_dir}/best_model.pth")
        print(f"Logs saved to: {args.log_dir}")


if __name__ == '__main__':
    main()
