import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from snn.models import EventSNNFlowNetLite
from snn.dataset import OpticalFlowDataset
from snn.training import SNNTrainer
from snn.utils.logger import Logger



def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model(config: dict) -> torch.nn.Module:
    model_type = config.get('model_type', 'EventSNNFlowNetLite')

    if model_type == 'EventSNNFlowNetLite':
        model = EventSNNFlowNetLite(
            config=config,
        )

    return model


def build_model(config: dict, device='cuda', train=True, checkpoint_path=None) -> torch.nn.Module:
    """Build model from configuration"""
    
    if train:

        model = get_model(config)

        model = model.to(device)
    
        return model
    else:
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
        config = checkpoint.get('config', {})

        model = get_model(config)
    
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
    
        print(f"Loaded model from {checkpoint_path}")
        print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Best validation EPE: {checkpoint.get('best_val_epe', 'unknown')}")
    
        return model, config
