"""
Debug script to check what the model is actually predicting
"""

import torch
import numpy as np
import h5py
from pathlib import Path
import yaml

import sys 
sys.path.insert(0, '..')
from snn.data.optical_flow_dataset import OpticalFlowDataset
from snn.models import SpikingFlowNet, SpikingFlowNetLite
import matplotlib.pyplot as plt


def load_model(checkpoint_path, config_path):
    """Load model from checkpoint"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model based on config
    model_type = config.get('model_type', 'SpikingFlowNetLite')
    if model_type == 'SpikingFlowNet':
        model = SpikingFlowNet(
            in_channels=config.get('in_channels', 5),
            num_timesteps=config.get('num_timesteps', 6),
            tau=config.get('tau', 2.0),
            threshold=config.get('threshold', 1.0)
        )
    elif model_type == 'SpikingFlowNetLite':
        model = SpikingFlowNetLite(
            in_channels=config.get('in_channels', 5),
            num_timesteps=config.get('num_timesteps', 6),
            tau=config.get('tau', 2.0),
            threshold=config.get('threshold', 1.0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    return model, config


def analyze_sample(model, sample, device='cpu'):
    """Analyze what the model predicts"""
    model = model.to(device)
    
    # Get input and ground truth
    input_tensor = sample['input'].unsqueeze(0).to(device)
    flow_gt = sample['flow'].numpy()  # [2, H, W]
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict):
            flow_pred = output['flow'].squeeze(0).cpu().numpy()
        else:
            flow_pred = output.squeeze(0).cpu().numpy()
    
    # Print statistics
    print("\n" + "="*60)
    print("Ground Truth Flow Statistics:")
    print(f"  Shape: {flow_gt.shape}")
    print(f"  U component: min={flow_gt[0].min():.4f}, max={flow_gt[0].max():.4f}, mean={flow_gt[0].mean():.4f}")
    print(f"  V component: min={flow_gt[1].min():.4f}, max={flow_gt[1].max():.4f}, mean={flow_gt[1].mean():.4f}")
    mag_gt = np.sqrt(flow_gt[0]**2 + flow_gt[1]**2)
    print(f"  Magnitude: min={mag_gt.min():.4f}, max={mag_gt.max():.4f}, mean={mag_gt.mean():.4f}")
    
    print("\nPredicted Flow Statistics:")
    print(f"  Shape: {flow_pred.shape}")
    print(f"  U component: min={flow_pred[0].min():.4f}, max={flow_pred[0].max():.4f}, mean={flow_pred[0].mean():.4f}")
    print(f"  V component: min={flow_pred[1].min():.4f}, max={flow_pred[1].max():.4f}, mean={flow_pred[1].mean():.4f}")
    mag_pred = np.sqrt(flow_pred[0]**2 + flow_pred[1]**2)
    print(f"  Magnitude: min={mag_pred.min():.4f}, max={mag_pred.max():.4f}, mean={mag_pred.mean():.4f}")
    
    # EPE
    epe = np.sqrt((flow_gt[0] - flow_pred[0])**2 + (flow_gt[1] - flow_pred[1])**2)
    print(f"\nEndpoint Error (EPE): {epe.mean():.4f}")
    
    # Check if prediction is near zero (bad)
    if mag_pred.max() < 0.1:
        print("\n⚠️  WARNING: Predicted flow magnitude is very small!")
        print("    The model might not have learned anything meaningful.")
    
    # Check correlation
    corr_u = np.corrcoef(flow_gt[0].flatten(), flow_pred[0].flatten())[0, 1]
    corr_v = np.corrcoef(flow_gt[1].flatten(), flow_pred[1].flatten())[0, 1]
    print(f"\nCorrelation:")
    print(f"  U component: {corr_u:.4f}")
    print(f"  V component: {corr_v:.4f}")
    
    if corr_u < 0 or corr_v < 0:
        print("\n⚠️  WARNING: Negative correlation detected!")
        print("    The model might be predicting inverted flow.")
    
    print("="*60 + "\n")
    
    return flow_gt, flow_pred


def main():
    # Paths
    checkpoint = '../checkpoints/best_model.pth'
    config = '../snn/configs/lightweight.yaml'
    data_dir = '../../blink_sim/output/train_girl1'
    sample = "girl1_Running_0"
    
    print("Loading model...")
    model, cfg = load_model(checkpoint, config)
    
    # Get number of event bins (use num_bins if specified, otherwise fall back to in_channels)
    num_event_bins = cfg.get('num_bins', cfg.get('in_channels', 5))
    
    print("Loading dataset...")
    dataset = OpticalFlowDataset(
        data_root=data_dir,
        split='val',
        num_bins=num_event_bins,
        crop_size=(cfg.get('height', 256), cfg.get('width', 256))
    )
    
    # Get a sample from girl1_BaseballHit_0 (middle of action)
    print(f"Finding {sample} sequence...")
    target_idx = None
    for idx in range(len(dataset)):
        sample_path = str(dataset.samples[idx])
        if sample in sample_path:
            # Extract frame number from path
            if '000020' in sample_path or '00020' in sample_path:  # Frame 20 - middle of action
                target_idx = idx
                break
    
    if target_idx is None:
        # Just get a middle frame from the sequence
        for idx in range(len(dataset)):
            if sample in str(dataset.samples[idx]):
                if idx >= 20 and idx < len(dataset):  # Skip first frames
                    target_idx = idx
                    break
    
    if target_idx is None:
        print("Could not find suitable frame, using index 20")
        target_idx = 20
    
    sample = dataset[target_idx]
    print(f"Using index {target_idx}")
    print(f"  Sample info: {dataset.samples[target_idx]}")
    
    # Analyze the prediction
    flow_gt, flow_pred = analyze_sample(model, sample)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Ground truth magnitude
    mag_gt = np.sqrt(flow_gt[0]**2 + flow_gt[1]**2)
    im0 = axes[0, 0].imshow(mag_gt, cmap='jet')
    axes[0, 0].set_title('Ground Truth Magnitude')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Predicted magnitude
    mag_pred = np.sqrt(flow_pred[0]**2 + flow_pred[1]**2)
    im1 = axes[0, 1].imshow(mag_pred, cmap='jet')
    axes[0, 1].set_title('Predicted Magnitude')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Error
    error = np.sqrt((flow_gt[0] - flow_pred[0])**2 + (flow_gt[1] - flow_pred[1])**2)
    im2 = axes[1, 0].imshow(error, cmap='hot')
    axes[1, 0].set_title('Endpoint Error')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Histogram
    axes[1, 1].hist(mag_gt.flatten(), bins=50, alpha=0.5, label='Ground Truth', density=True)
    axes[1, 1].hist(mag_pred.flatten(), bins=50, alpha=0.5, label='Predicted', density=True)
    axes[1, 1].set_xlabel('Flow Magnitude')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Magnitude Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_outputs/debug_flow_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to debug_outputs/debug_flow_analysis.png")


if __name__ == '__main__':
    main()
