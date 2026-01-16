"""
Main training script for SNN Optical Flow
"""

import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from snn.models import EventSNNFlowNetLite, EventSNNFlowNetLiteV2
from snn.data import OpticalFlowDataset
from snn.data.data_utils import Compose, RandomHorizontalFlip, RandomCrop, Normalize
from snn.training import SNNTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train SNN for Optical Flow')
    
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--data-root', type=str, default='../blink_sim/output',
                      help='Root directory for dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                      help='Directory for logs')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: dict, logger=None) -> torch.nn.Module:
    """Build model from configuration"""
    model_type = config.get('model_type', 'SpikingFlowNet')
    
    # Enable quantization logging if quantization is enabled
    enable_quant_logging = config.get('quantization_enabled', False) and config.get('log_quantization', False)
    
    if model_type == 'EventSNNFlowNetLite':
        # EventSNNFlowNetLite uses different parameters
        model = EventSNNFlowNetLite(
            base_ch=config.get('base_ch', 32),
            tau=config.get('tau', 2.0),
            threshold=config.get('threshold', 1.0),
            alpha=config.get('alpha', 10.0),
            use_bn=config.get('use_bn', False),
            quantize=config.get('quantization_enabled', False),
            weight_bit_width=config.get('weight_bit_width', 8),
            act_bit_width=config.get('act_bit_width', 8),
            binarize=config.get('binarize', False)
        )
    elif model_type == 'EventSNNFlowNetLiteV2':
        model = EventSNNFlowNetLiteV2(
            base_ch=config.get('base_ch', 32),
            tau=config.get('tau', 2.0),
            threshold=config.get('threshold', 1.0),
            alpha=config.get('alpha', 10.0),
            use_bn=config.get('use_bn', False),
            quantize=config.get('quantization_enabled', False),
            weight_bit_width=config.get('weight_bit_width', 8),
            act_bit_width=config.get('act_bit_width', 8),
            binarize=config.get('binarize', False),
            hardware_mode=config.get('hardware_mode', False),
            output_bit_width=config.get('output_bit_width', 16),
            first_layer_bit_width=config.get('first_layer_bit_width', 8),
            mem_bit_width=config.get('mem_bit_width', 16),
            enable_logging=enable_quant_logging,
            logger=logger
        )
    
    return model


def build_dataloaders(config: dict, data_root: str = None):
    """Build train and validation dataloaders"""
    
    # Use data_root from config if not provided as argument
    if data_root is None:
        data_root = config.get('data_root', '../blink_sim/output')
    
    # Data transforms
    train_transform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomCrop(crop_size=config.get('crop_size', (320, 320))),
        Normalize()
    ])
    
    val_transform = Compose([
        # Validation uses center crop (no random cropping)
        # Note: Dataset will apply _center_crop if no crop transform is present
        Normalize()
    ])
    
    # Get number of event bins (use num_bins if specified, otherwise fall back to in_channels)
    num_event_bins = config.get('num_bins', config.get('in_channels', 5))
    
    # Datasets
    train_dataset = OpticalFlowDataset(
        data_root=data_root,
        split='train',
        transform=None,  # Let dataset handle cropping for consistency
        use_events=config.get('use_events', True),
        num_bins=num_event_bins,
        data_size=config.get('data_size', (320, 320)),
        crop_size=config.get('crop_size', (320, 320)),
        max_samples=config.get('max_train_samples', None)
    )
    
    #val_dataset = OpticalFlowDataset(
    #    data_root=data_root,
    #    split='val',
    #    transform=None,  # Let dataset handle cropping for consistency
    #    use_events=config.get('use_events', True),
    #    num_bins=num_event_bins,
    #    crop_size=config.get('crop_size', (320, 320)),
    #    max_samples=config.get('max_val_samples', None)
    #)
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
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
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
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create logger first (needed for quantization logging)
    from snn.utils.logger import Logger
    logger = Logger(log_dir=args.log_dir)
    
    # Build model with logger for quantization statistics
    model = build_model(config, logger=logger)
    print(f"Built model: {config.get('model_type', 'SpikingFlowNet')}")
    
    # Log quantization status
    if config.get('quantization_enabled', False):
        print(f"Quantization enabled: W{config.get('weight_bit_width', 8)}A{config.get('act_bit_width', 8)}")
        if config.get('log_quantization', False):
            print(f"✓ Quantization statistics will be logged to TensorBoard")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Build dataloaders
    # Command line arg overrides config value
    data_root = config.get('data_root', args.data_root)
    train_loader, val_loader = build_dataloaders(config, data_root)
    print(f"Data root: {data_root}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create trainer (using the logger we already created)
    trainer = SNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        logger=logger  # Pass existing logger
    )
    
    # Log configuration
    trainer.logger.log_config(config)
    
    # Train
    num_epochs = config.get('num_epochs', 200)
    trainer.train(num_epochs=num_epochs, resume=args.resume)


if __name__ == '__main__':
    main()
