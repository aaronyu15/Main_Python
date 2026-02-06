import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from snn.models import EventSNNFlowNetLite
from snn.dataset import OpticalFlowDataset
from snn.training import SNNTrainer
from snn.utils.logger import Logger

from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train SNN for Optical Flow')
    
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--data-root', type=str, default='../blink_sim/output',
                      help='Root directory for dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                      help='Directory for logs')
    
    return parser.parse_args()


def build_dataloaders(config: dict, data_root: str = None):
    if data_root is None:
        data_root = config.get('data_root', '../blink_sim/output')
    
    # Create config copy with data_root override if provided
    dataset_config = config.copy()
    dataset_config['data_root'] = data_root
    
    train_dataset = OpticalFlowDataset(config=dataset_config)
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
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
    
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    
    model = build_model(config, device=device, train=True)
    print(f"Built model: {config.get('model_type', 'SpikingFlowNet')}")
    
    logger = Logger(log_dir=args.log_dir)
    model.set_logger(logger)
    
    if config.get('quantize_weights', False) or config.get('quantize_activations', False) or config.get('quantize_mem', False):
        print(f"Quantization enabled: W{config.get('weight_bit_width', 8)}A{config.get('act_bit_width', 8)}M{config.get('mem_bit_width', 16)}")
        print(f"  Weight quantization: {config.get('quantize_weights', False)}")
        print(f"  Activation quantization: {config.get('quantize_activations', False)}")
        print(f"  Membrane quantization: {config.get('quantize_mem', False)}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    data_root = config.get('data_root', args.data_root)
    train_loader, val_loader = build_dataloaders(config, data_root)

    print(f"Data root: {data_root}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
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
    
    # Train
    num_epochs = config.get('num_epochs', 100)
    trainer.train(num_epochs=num_epochs, resume=args.resume)


if __name__ == '__main__':
    main()
