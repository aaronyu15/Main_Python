"""
Debug script to visualize dataset samples
Checks if OpticalFlowDataset is producing correct input tensors and labels
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

import sys 
sys.path.insert(0, '..')
from snn.data.optical_flow_dataset import OpticalFlowDataset
from snn.utils.visualization import flow_to_color


def visualize_event_bins(event_tensor, title="Event Bins"):
    """
    Visualize the event representation tensor
    
    Args:
        event_tensor: Tensor of shape [num_bins, H, W] or [H, W, num_bins]
    """
    if event_tensor.ndim == 3:
        if event_tensor.shape[0] < event_tensor.shape[2]:
            # [num_bins, H, W] -> [H, W, num_bins]
            event_tensor = event_tensor.permute(1, 2, 0)
    
    num_bins = event_tensor.shape[2]
    
    fig, axes = plt.subplots(1, num_bins + 1, figsize=(3 * (num_bins + 1), 3))
    
    # Show each temporal bin
    for i in range(num_bins):
        bin_data = event_tensor[:, :, i].numpy()
        im = axes[i].imshow(bin_data, cmap='gray')
        axes[i].set_title(f'Bin {i}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    # Show composite (sum of all bins)
    composite = event_tensor.sum(dim=2).numpy()
    im = axes[num_bins].imshow(composite, cmap='hot')
    axes[num_bins].set_title('Composite (Sum)')
    axes[num_bins].axis('off')
    plt.colorbar(im, ax=axes[num_bins], fraction=0.046)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def visualize_sample(sample, sample_idx, sequence_name=None):
    """
    Visualize a complete dataset sample
    
    Args:
        sample: Dictionary containing 'input', 'flow', 'valid_mask'
        sample_idx: Index of the sample
        sequence_name: Optional sequence name for title
    """
    input_tensor = sample['input']  # [num_bins, H, W]
    flow_gt = sample['flow']        # [2, H, W]
    valid_mask = sample['valid_mask']  # [1, H, W]
    
    print("="*80)
    print(f"Sample {sample_idx}" + (f" from {sequence_name}" if sequence_name else ""))
    print("="*80)
    
    # Print tensor statistics
    print("\nInput Tensor:")
    print(f"  Shape: {input_tensor.shape}")
    print(f"  Dtype: {input_tensor.dtype}")
    print(f"  Range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
    print(f"  Mean: {input_tensor.mean():.4f}, Std: {input_tensor.std():.4f}")
    
    for i in range(input_tensor.shape[0]):
        bin_data = input_tensor[i]
        print(f"  Bin {i}: min={bin_data.min():.4f}, max={bin_data.max():.4f}, "
              f"mean={bin_data.mean():.4f}, nonzero={(bin_data != 0).sum().item()}")
    
    print("\nGround Truth Flow:")
    print(f"  Shape: {flow_gt.shape}")
    print(f"  Dtype: {flow_gt.dtype}")
    print(f"  U component: min={flow_gt[0].min():.4f}, max={flow_gt[0].max():.4f}, mean={flow_gt[0].mean():.4f}")
    print(f"  V component: min={flow_gt[1].min():.4f}, max={flow_gt[1].max():.4f}, mean={flow_gt[1].mean():.4f}")
    
    flow_mag = torch.sqrt(flow_gt[0]**2 + flow_gt[1]**2)
    print(f"  Magnitude: min={flow_mag.min():.4f}, max={flow_mag.max():.4f}, mean={flow_mag.mean():.4f}")
    
    print("\nValid Mask:")
    print(f"  Shape: {valid_mask.shape}")
    print(f"  Valid pixels: {valid_mask.sum().item()} / {valid_mask.numel()} ({100*valid_mask.float().mean():.1f}%)")
    
    # Visualize event bins
    fig1 = visualize_event_bins(input_tensor.permute(1, 2, 0), 
                                title=f"Event Representation - Sample {sample_idx}")
    
    # Visualize flow and components
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Flow components
    im0 = axes[0, 0].imshow(flow_gt[0].numpy(), cmap='RdBu', vmin=-flow_mag.max(), vmax=flow_mag.max())
    axes[0, 0].set_title('Flow U (horizontal)')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(flow_gt[1].numpy(), cmap='RdBu', vmin=-flow_mag.max(), vmax=flow_mag.max())
    axes[0, 1].set_title('Flow V (vertical)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Flow magnitude
    im2 = axes[0, 2].imshow(flow_mag.numpy(), cmap='hot')
    axes[0, 2].set_title('Flow Magnitude')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Flow color visualization (Middlebury)
    flow_np = flow_gt.numpy().transpose(1, 2, 0)  # [H, W, 2]
    flow_color = flow_to_color(flow_np)
    axes[1, 0].imshow(flow_color)
    axes[1, 0].set_title('Flow Visualization (Middlebury)')
    axes[1, 0].axis('off')
    
    # Valid mask
    axes[1, 1].imshow(valid_mask[0].numpy(), cmap='gray')
    axes[1, 1].set_title('Valid Mask')
    axes[1, 1].axis('off')
    
    # Event composite
    event_composite = input_tensor.sum(dim=0).numpy()
    im5 = axes[1, 2].imshow(event_composite, cmap='hot')
    axes[1, 2].set_title('Event Composite')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2])
    
    title = f"Ground Truth Flow - Sample {sample_idx}"
    if sequence_name:
        title += f" ({sequence_name})"
    fig2.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig1, fig2


def main():
    # Configuration
    config_path = '../snn/configs/lightweight.yaml'
    data_root = '../../blink_sim/output/train_girl1'
    
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get number of event bins (use num_bins if specified, otherwise fall back to in_channels)
    num_event_bins = config.get('num_bins', config.get('in_channels', 5))
    
    # Create dataset
    print(f"Loading dataset from {data_root}...")
    print(f"Using {num_event_bins} event temporal bins")
    dataset = OpticalFlowDataset(
        data_root=data_root,
        split='train',
        num_bins=num_event_bins,
        crop_size=tuple(config.get('crop_size', [224, 224]))
    )
    
    print(f"Dataset loaded: {len(dataset)} samples from {len(dataset.sequences)} sequences")
    
    # List available sequences
    print("\nAvailable sequences:")
    sequences = {}
    for idx, sample_info in enumerate(dataset.samples):
        seq_name = sample_info['sequence']
        if seq_name not in sequences:
            sequences[seq_name] = []
        sequences[seq_name].append(idx)
    
    for seq_name in sorted(sequences.keys())[:10]:  # Show first 10
        print(f"  {seq_name}: {len(sequences[seq_name])} frames")
    print(f"  ... and {len(sequences) - 10} more sequences" if len(sequences) > 10 else "")
    
    # Get a sample from a specific sequence
    target_sequence = 'girl1_Running_0'
    print(f"\nLooking for sequence: {target_sequence}")
    
    if target_sequence in sequences:
        # Get a frame from the middle of the sequence (where there's motion)
        seq_indices = sequences[target_sequence]
        sample_idx = seq_indices[min(20, len(seq_indices) // 2)]
        print(f"Using sample index {sample_idx} (frame {sample_idx - seq_indices[0]} of {len(seq_indices)})")
    else:
        # Just use a sample from the middle of the dataset
        sample_idx = len(dataset) // 2
        target_sequence = dataset.samples[sample_idx]['sequence']
        print(f"Sequence not found, using sample {sample_idx} from {target_sequence}")
    
    # Get the sample
    print(f"\nLoading sample {sample_idx}...")
    sample = dataset[sample_idx]
    
    # Visualize
    print("\nCreating visualizations...")
    fig1, fig2 = visualize_sample(sample, sample_idx, target_sequence)
    
    # Save figures
    output_dir = Path('debug_outputs')
    output_dir.mkdir(exist_ok=True)
    
    fig1_path = output_dir / f'sample_{sample_idx:04d}_events.png'
    fig2_path = output_dir / f'sample_{sample_idx:04d}_flow.png'
    
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    
    print(f"\nSaved visualizations:")
    print(f"  Events: {fig1_path}")
    print(f"  Flow: {fig2_path}")
    
    # Show additional samples for comparison
    print("\n" + "="*80)
    print("Additional samples for comparison")
    print("="*80)
    
    # Get a few more samples from different parts of the sequence
    additional_indices = []
    if target_sequence in sequences:
        seq_indices = sequences[target_sequence]
        # Get samples from start, quarter, middle, three-quarters
        for fraction in [0.0, 0.25, 0.5, 0.75]:
            idx = seq_indices[int(len(seq_indices) * fraction)]
            if idx != sample_idx:
                additional_indices.append(idx)
    
    for idx in additional_indices[:3]:  # Limit to 3 additional samples
        sample = dataset[idx]
        print(f"\n--- Sample {idx} ---")
        print(f"Input range: [{sample['input'].min():.4f}, {sample['input'].max():.4f}]")
        print(f"Input nonzero bins: {[(sample['input'][i] != 0).sum().item() for i in range(sample['input'].shape[0])]}")
        flow_mag = torch.sqrt(sample['flow'][0]**2 + sample['flow'][1]**2)
        print(f"Flow magnitude: min={flow_mag.min():.4f}, max={flow_mag.max():.4f}, mean={flow_mag.mean():.4f}")
    
    print("\n" + "="*80)
    print("Dataset debugging complete!")
    print("="*80)
    plt.show()


if __name__ == '__main__':
    main()
