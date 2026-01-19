"""
Quantization-Aware Fine-tuning Script for EventSNNFlowNetLite

This script enables you to:
1. Load a pre-trained full-precision model from checkpoints/V2/best_model.pth
2. Enable quantization-aware training (QAT) with specified bit-width
3. Fine-tune the model to adapt to quantization constraints

Usage:
    # Fine-tune with 8-bit quantization
    python finetune_quantized.py --config snn/configs/event_snn_lite_8bit.yaml \
                                  --pretrained checkpoints/V2/best_model.pth \
                                  --checkpoint-dir checkpoints/V2_8bit \
                                  --log-dir logs/V2_8bit

    # Fine-tune with 4-bit quantization (recommended to start from 8-bit checkpoint)
    python finetune_quantized.py --config snn/configs/event_snn_lite_4bit.yaml \
                                  --pretrained checkpoints/V2_8bit/best_model.pth \
                                  --checkpoint-dir checkpoints/V2_4bit \
                                  --log-dir logs/V2_4bit

Progressive quantization workflow:
    1. Full-precision (32-bit) → 8-bit
    2. 8-bit → 4-bit
    3. 4-bit → 2-bit
    4. 2-bit → 1-bit (binary)
"""

import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from snn.models import EventSNNFlowNetLite
from snn.data import OpticalFlowDataset
from snn.data.data_utils import Compose, RandomHorizontalFlip, RandomCrop, Normalize
from snn.training import SNNTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Quantization-Aware Fine-tuning for EventSNNFlowNetLite')
    
    parser.add_argument('--config', type=str, required=True,
                      help='Path to quantization configuration file (e.g., snn/configs/event_snn_lite_8bit.yaml)')
    parser.add_argument('--pretrained', type=str, required=True,
                      help='Path to pre-trained checkpoint to fine-tune from (e.g., checkpoints/V2/best_model.pth)')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--data-root', type=str, default=None,
                      help='Root directory for dataset (overrides config)')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                      help='Directory to save quantized checkpoints')
    parser.add_argument('--log-dir', type=str, required=True,
                      help='Directory for logs')
    parser.add_argument('--strict-load', action='store_true',
                      help='Use strict loading (fails if keys mismatch). Default: False for flexibility')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_quantized_model(config: dict, logger=None) -> torch.nn.Module:
    """Build quantized model from configuration"""
    model_type = config.get('model_type', 'EventSNNFlowNetLite')
    
    if model_type != 'EventSNNFlowNetLite':
        raise ValueError(f"This script only supports EventSNNFlowNetLite, got {model_type}")
    
    # Enable parameter logging (default to True for quantized fine-tuning)
    log_params = config.get('log_params', True)
    
    # Build model with quantization enabled
    model = EventSNNFlowNetLite(
        base_ch=config.get('base_ch', 32),
        tau=config.get('tau', 2.0),
        threshold=config.get('threshold', 1.0),
        alpha=config.get('alpha', 10.0),
        use_bn=config.get('use_bn', False),
        quantize_weights=config.get('quantize_weights', True),
        quantize_activations=config.get('quantize_activations', True),
        quantize_mem=config.get('quantize_mem', True),
        weight_bit_width=config.get('weight_bit_width', 8),
        act_bit_width=config.get('act_bit_width', 8),
        binarize=config.get('binarize', False),
        hardware_mode=config.get('hardware_mode', False),
        output_bit_width=config.get('output_bit_width', 16),
        first_layer_bit_width=config.get('first_layer_bit_width', 8),
        mem_bit_width=config.get('mem_bit_width', 16),
        enable_logging=log_params,
        logger=logger
    )
    
    return model


def load_pretrained_weights(model: torch.nn.Module, checkpoint_path: str, 
                           device: str = 'cuda', strict: bool = False):
    """
    Load pre-trained weights into quantized model
    
    Args:
        model: Quantized model (with quantization layers)
        checkpoint_path: Path to pre-trained checkpoint
        device: Device to load to
        strict: Whether to use strict loading (set False for flexibility)
    
    Returns:
        Loaded model
    """
    print(f"\nLoading pre-trained weights from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict (handle both direct state_dict and checkpoint dict)
    if 'model_state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['model_state_dict']
        print(f"  Loaded from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        pretrained_state_dict = checkpoint
    
    # Get model state dict to compare
    model_state = model.state_dict()
    total_model_keys = len(model_state)
    total_checkpoint_keys = len(pretrained_state_dict)
    
    print(f"  Model expects {total_model_keys} parameters")
    print(f"  Checkpoint contains {total_checkpoint_keys} parameters")
    
    # Load weights (non-strict to allow quantization layer mismatches)
    missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=strict)
    
    if not strict:
        # Filter out expected missing keys (quantization layers)
        quant_missing = [k for k in missing_keys if 'act_quant' in k]
        other_missing = [k for k in missing_keys if k not in quant_missing]
        
        # Count successfully loaded keys
        loaded_keys = total_model_keys - len(missing_keys)
        
        print(f"\n  Weight Loading Summary:")
        print(f"  {'='*60}")
        print(f"  ✓ Successfully loaded: {loaded_keys}/{total_model_keys} parameters")
        print(f"  ↻ Quantization layers (initialized randomly): {len(quant_missing)}")
        
        if len(quant_missing) > 0:
            # Show breakdown of quantization layers by type
            running_min = len([k for k in quant_missing if 'running_min' in k])
            running_max = len([k for k in quant_missing if 'running_max' in k])
            num_batches = len([k for k in quant_missing if 'num_batches_tracked' in k])
            print(f"      • Running statistics (min/max): {running_min + running_max} buffers")
            print(f"      • Batch counters: {num_batches} buffers")
            print(f"      These will be learned during quantization-aware training")
        
        if other_missing:
            print(f"\n  ⚠ WARNING: Missing non-quantization keys: {len(other_missing)}")
            print(f"      This may indicate a model architecture mismatch!")
            for key in other_missing[:10]:  # Show first 10
                print(f"      • {key}")
            if len(other_missing) > 10:
                print(f"      ... and {len(other_missing) - 10} more")
        
        if unexpected_keys:
            print(f"\n  ⚠ WARNING: Unexpected keys in checkpoint: {len(unexpected_keys)}")
            print(f"      These weights exist in checkpoint but not in model:")
            for key in unexpected_keys[:10]:  # Show first 10
                print(f"      • {key}")
            if len(unexpected_keys) > 10:
                print(f"      ... and {len(unexpected_keys) - 10} more")
        
        print(f"  {'='*60}")
        
        if len(other_missing) == 0 and len(unexpected_keys) == 0:
            print(f"  ✓ All convolution weights transferred successfully!")
            print(f"  ✓ Model is ready for quantization-aware fine-tuning")
        else:
            print(f"  ⚠ Please review warnings above before proceeding")
    
    return model


def build_dataloaders(config: dict, data_root: str = None):
    """Build train and validation dataloaders"""
    
    # Use data_root from config if not provided as argument
    if data_root is None:
        data_root = config.get('data_root', '../blink_sim/output/train')
    
    # Get number of event bins
    num_event_bins = config.get('num_bins', config.get('in_channels', 5))
    
    # Datasets
    train_dataset = OpticalFlowDataset(
        data_root=data_root,
        split='train',
        transform=None,
        use_events=config.get('use_events', True),
        num_bins=num_event_bins,
        data_size=config.get('data_size', (320, 320)),
        crop_size=config.get('crop_size', (320, 320)),
        max_samples=config.get('max_train_samples', None)
    )
    
    val_dataset = []
    
    # If no validation set, use a portion of training
    if len(val_dataset) == 0:
        print("No validation set found, splitting training set...")
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 2),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', 2),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"=" * 80)
    print(f"Quantization-Aware Fine-tuning")
    print(f"=" * 80)
    print(f"Config: {args.config}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Weight bit-width: {config.get('weight_bit_width', 8)}-bit")
    print(f"Activation bit-width: {config.get('act_bit_width', 8)}-bit")
    print(f"Membrane bit-width: {config.get('mem_bit_width', 16)}-bit")
    print(f"Weight quantization: {config.get('quantize_weights', True)}")
    print(f"Activation quantization: {config.get('quantize_activations', True)}")
    print(f"Membrane quantization: {config.get('quantize_mem', True)}")
    print(f"Binarize: {config.get('binarize', False)}")
    if config.get('log_params', True):
        print(f"✓ Model parameters and statistics will be logged to TensorBoard")
    print(f"=" * 80)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create logger first (needed for quantization logging)
    from snn.utils.logger import Logger
    logger = Logger(log_dir=args.log_dir)
    
    # Build quantized model with logger
    print("\nBuilding quantized model...")
    model = build_quantized_model(config, logger=logger)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Load pre-trained weights
    model = load_pretrained_weights(model, args.pretrained, device, strict=args.strict_load)
    
    # Build dataloaders
    data_root = args.data_root if args.data_root else config.get('data_root')
    train_loader, val_loader = build_dataloaders(config, data_root)
    print(f"\nData root: {data_root}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer (using the logger we already created)
    print(f"\nInitializing trainer...")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Log dir: {log_dir}")
    
    trainer = SNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        logger=logger  # Pass existing logger
    )
    
    # Log configuration
    trainer.logger.log_config(config)
    
    # Start fine-tuning
    num_epochs = config.get('num_epochs', 30)
    print(f"\nStarting quantization-aware fine-tuning for {num_epochs} epochs...")
    print(f"Learning rate: {config.get('learning_rate', 0.0001)}")
    print(f"=" * 80)
    
    trainer.train(num_epochs=num_epochs)
    
    print(f"\n" + "=" * 80)
    print(f"Fine-tuning completed!")
    print(f"Best validation EPE: {trainer.best_val_epe:.4f}")
    print(f"Final checkpoint saved to: {checkpoint_dir / 'best_model.pth'}")
    print(f"=" * 80)


if __name__ == '__main__':
    main()
