"""
Visualization script for testing trained SNN optical flow models

This script loads a trained model, runs inference on dataset samples,
and creates visualizations comparing ground truth with predictions.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Dict, Tuple, Optional
import yaml

import sys 
sys.path.insert(0, '..')
from snn.models import SpikingFlowNetLite, EventSNNFlowNetLite
from snn.data import OpticalFlowDataset
from snn.utils.visualization import visualize_flow, flow_to_color


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: dict,
    device: torch.device
) -> torch.nn.Module:
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration dictionary
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    # Build model architecture
    model_type = config.get('model_type', 'SpikingFlowNet')
    
    if model_type == 'EventSNNFlowNetLite':
        model = EventSNNFlowNetLite(
            base_ch=config.get('base_ch', 32),
            tau=config.get('tau', 2.0),
            threshold=config.get('threshold', 1.0),
            alpha=config.get('alpha', 10.0),
            use_bn=config.get('use_bn', False),
            quantize=config.get('quantization_enabled', False),
            bit_width=config.get('initial_bit_width', 8),
            binarize=config.get('binarize', False)
        )
    else:
        # SpikingFlowNet and SpikingFlowNetLite both use SpikingFlowNetLite
        model_params = {
            'in_channels': config.get('in_channels', 5),
            'num_timesteps': config.get('num_timesteps', 10),
            'tau': config.get('tau', 2.0),
            'threshold': config.get('threshold', 1.0),
            'binarize': config.get('binarize', False),
            'quantize': config.get('quantization_enabled', False),
            'bit_width': config.get('initial_bit_width', 4 if model_type == 'SpikingFlowNetLite' else 32),
        }
        model = SpikingFlowNetLite(**model_params)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        val_epe = checkpoint.get('val_epe', None)
        if val_epe is not None:
            print(f"Validation EPE: {val_epe:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def load_dataset(config: dict, data_root: Optional[str] = None) -> OpticalFlowDataset:
    """
    Load dataset for inference
    
    Args:
        config: Configuration dictionary
        data_root: Override data root path
    
    Returns:
        Dataset instance
    """
    
    # Get number of event bins (use num_bins if specified, otherwise fall back to in_channels)
    num_event_bins = config.get('num_bins', config.get('in_channels', 5))
    
    dataset = OpticalFlowDataset(
        data_root=data_root,
        split='train',  # Can visualize train or val samples
        transform=None,  # No augmentation for visualization
        use_events=config.get('use_events', True),
        num_bins=num_event_bins,
        crop_size=config.get('crop_size', (256, 256)),
        max_samples=None  # Load all samples
    )
    
    return dataset


def get_sequences(dataset: OpticalFlowDataset) -> Dict[str, list]:
    """
    Get all available sequences and their sample indices
    
    Args:
        dataset: Dataset instance
    
    Returns:
        Dictionary mapping sequence names to lists of sample indices
    """
    sequences = {}
    for idx, sample_info in enumerate(dataset.samples):
        seq_name = sample_info['sequence']
        if seq_name not in sequences:
            sequences[seq_name] = []
        sequences[seq_name].append(idx)
    
    return sequences


def list_sequences(dataset: OpticalFlowDataset):
    """
    Print all available sequences with their frame counts
    
    Args:
        dataset: Dataset instance
    """
    sequences = get_sequences(dataset)
    
    print("\nAvailable Sequences:")
    print("-" * 60)
    for seq_name in sorted(sequences.keys()):
        frame_count = len(sequences[seq_name])
        print(f"  {seq_name:40s} - {frame_count:3d} frames")
    print("-" * 60)
    print(f"Total: {len(sequences)} sequences, {len(dataset)} frames\n")


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[torch.Tensor, Dict]:
    """
    Run inference on a single sample
    
    Args:
        model: Trained model
        sample: Sample dictionary from dataset
        device: Device to run on
    
    Returns:
        Tuple of (predicted_flow, metrics_dict)
    """
    # Move input to device and add batch dimension
    input_tensor = sample['input'].unsqueeze(0).to(device)
    flow_gt = sample['flow']
    
    # Run inference
    output = model(input_tensor)
    
    # Get predicted flow (remove batch dimension)
    if isinstance(output, dict):
        flow_pred = output['flow'].squeeze(0).cpu()
    else:
        flow_pred = output.squeeze(0).cpu()
    
    # Compute metrics
    epe = torch.norm(flow_pred - flow_gt, p=2, dim=0).mean()
    
    metrics = {
        'epe': epe.item(),
        'max_flow_gt': torch.norm(flow_gt, p=2, dim=0).max().item(),
        'max_flow_pred': torch.norm(flow_pred, p=2, dim=0).max().item(),
    }
    
    return flow_pred, metrics


def visualize_flow_comparison(
    input_events: torch.Tensor,
    flow_gt: torch.Tensor,
    flow_pred: torch.Tensor,
    metrics: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Create side-by-side visualization of ground truth and predicted flow
    
    Args:
        input_events: Input event voxel grid [C, H, W]
        flow_gt: Ground truth flow [2, H, W]
        flow_pred: Predicted flow [2, H, W]
        metrics: Dictionary of metrics
        save_path: Optional path to save figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert flows to numpy
    if isinstance(flow_gt, torch.Tensor):
        flow_gt = flow_gt.cpu().numpy()
    if isinstance(flow_pred, torch.Tensor):
        flow_pred = flow_pred.cpu().numpy()
    if isinstance(input_events, torch.Tensor):
        input_events = input_events.cpu().numpy()
    
    # Compute max flow for consistent visualization
    max_flow = max(
        np.sqrt((flow_gt**2).sum(axis=0)).max(),
        np.sqrt((flow_pred**2).sum(axis=0)).max(),
        1.0
    )
    
    # Row 1: Input visualization
    # Show events by polarity (red=positive, blue=negative)
    # Handle both old [num_bins, H, W] and new [num_bins, 2, H, W] formats
    if input_events.ndim == 4:  # [num_bins, 2, H, W] - polarity-separated
        # Sum across time bins: [num_bins, 2, H, W] -> [2, H, W]
        event_sum = input_events.sum(axis=0)  # [2, H, W]
        pos_events = event_sum[0]  # Positive events
        neg_events = event_sum[1]  # Negative events
    else:  # [num_bins, H, W] - old voxel grid format
        event_sum = input_events.sum(axis=0)  # [H, W]
        pos_events = np.maximum(event_sum, 0)
        neg_events = np.maximum(-event_sum, 0)
    
    h, w = pos_events.shape
    event_rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # Positive events -> red channel
    if pos_events.max() > 0:
        event_rgb[:, :, 0] = pos_events / (pos_events.max() * 0.5)
    
    # Negative events -> blue channel
    if neg_events.max() > 0:
        event_rgb[:, :, 2] = neg_events / (neg_events.max() * 0.5)
    
    event_rgb = np.clip(event_rgb, 0, 1)
    axes[0, 0].imshow(event_rgb)
    axes[0, 0].set_title('Input Events (red=pos, blue=neg)')
    axes[0, 0].axis('off')
    
    # Ground truth flow
    flow_gt_color = flow_to_color(flow_gt, max_flow)
    axes[0, 1].imshow(flow_gt_color)
    axes[0, 1].set_title('Ground Truth Flow')
    axes[0, 1].axis('off')
    
    # Predicted flow
    flow_pred_color = flow_to_color(flow_pred, max_flow)
    axes[0, 2].imshow(flow_pred_color)
    axes[0, 2].set_title('Predicted Flow')
    axes[0, 2].axis('off')
    
    # Row 2: Error visualization and quiver plots
    # Flow error
    flow_error = np.sqrt(((flow_gt - flow_pred)**2).sum(axis=0))
    im = axes[1, 0].imshow(flow_error, cmap='hot')
    axes[1, 0].set_title(f'Flow Error (EPE: {metrics["epe"]:.3f})')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Ground truth quiver
    h, w = flow_gt.shape[1:]
    step = max(h // 20, 1)
    y, x = np.mgrid[0:h:step, 0:w:step]
    u_gt = flow_gt[0, ::step, ::step]
    v_gt = flow_gt[1, ::step, ::step]
    axes[1, 1].quiver(x, y, u_gt, v_gt, scale=max_flow*5, color='blue', alpha=0.7)
    axes[1, 1].set_title('GT Flow Vectors')
    axes[1, 1].set_xlim(0, w)
    axes[1, 1].set_ylim(h, 0)
    axes[1, 1].axis('off')
    
    # Predicted quiver
    u_pred = flow_pred[0, ::step, ::step]
    v_pred = flow_pred[1, ::step, ::step]
    axes[1, 2].quiver(x, y, u_pred, v_pred, scale=max_flow*5, color='red', alpha=0.7)
    axes[1, 2].set_title('Pred Flow Vectors')
    axes[1, 2].set_xlim(0, w)
    axes[1, 2].set_ylim(h, 0)
    axes[1, 2].axis('off')
    
    plt.suptitle(
        f'Optical Flow Prediction\n'
        f'EPE: {metrics["epe"]:.3f} | '
        f'Max Flow GT: {metrics["max_flow_gt"]:.2f} | '
        f'Max Flow Pred: {metrics["max_flow_pred"]:.2f}',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_flow_animation(
    model: torch.nn.Module,
    dataset: OpticalFlowDataset,
    indices: list,
    device: torch.device,
    save_path: str,
    fps: int = 5,
    sequence_name: Optional[str] = None
):
    """
    Create an animation showing predictions across multiple samples
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        indices: List of sample indices to visualize
        device: Device to run inference on
        save_path: Path to save animation (should end in .gif or .mp4)
        fps: Frames per second for animation
        sequence_name: Optional sequence name to display in title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    def update_frame(frame_idx):
        """Update function for animation"""
        idx = indices[frame_idx]
        sample = dataset[idx]
        
        # Run inference
        flow_pred, metrics = run_inference(model, sample, device)
        
        # Get data
        input_events = sample['input'].cpu().numpy()
        flow_gt = sample['flow'].cpu().numpy()
        flow_pred = flow_pred.cpu().numpy()
        metadata = sample['metadata']
        
        # Compute max flow for consistent coloring
        max_flow = max(
            np.sqrt((flow_gt**2).sum(axis=0)).max(),
            np.sqrt((flow_pred**2).sum(axis=0)).max(),
            1.0
        )
        
        # Clear axes
        for ax in axes:
            ax.clear()
        
        # Input events - visualize by polarity (red=positive, blue=negative)
        # Handle both old [num_bins, H, W] and new [num_bins, 2, H, W] formats
        if input_events.ndim == 4:  # [num_bins, 2, H, W] - polarity-separated
            # Sum across time bins: [num_bins, 2, H, W] -> [2, H, W]
            event_sum = input_events.sum(axis=0)  # [2, H, W]
            pos_events = event_sum[0]  # Positive events
            neg_events = event_sum[1]  # Negative events
        else:  # [num_bins, H, W] - old voxel grid format
            event_sum = input_events.sum(axis=0)  # [H, W]
            pos_events = np.maximum(event_sum, 0)
            neg_events = np.maximum(-event_sum, 0)
        
        # Create RGB image: red for positive, blue for negative
        h, w = pos_events.shape
        event_rgb = np.zeros((h, w, 3), dtype=np.float32)
        
        # Positive events -> red channel
        if pos_events.max() > 0:
            event_rgb[:, :, 0] = pos_events / (pos_events.max() * 0.5)  # Enhanced brightness
        
        # Negative events -> blue channel
        if neg_events.max() > 0:
            event_rgb[:, :, 2] = neg_events / (neg_events.max() * 0.5)  # Enhanced brightness
        
        event_rgb = np.clip(event_rgb, 0, 1)
        axes[0].imshow(event_rgb)
        axes[0].set_title(f'Events (Frame {metadata["index"]:03d})')
        axes[0].axis('off')
        
        # Ground truth
        flow_gt_color = flow_to_color(flow_gt, max_flow)
        axes[1].imshow(flow_gt_color)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        flow_pred_color = flow_to_color(flow_pred, max_flow)
        axes[2].imshow(flow_pred_color)
        axes[2].set_title(f'Prediction (EPE: {metrics["epe"]:.3f})')
        axes[2].axis('off')
        
        # Update title
        if sequence_name:
            title = f'{sequence_name} - Frame {metadata["index"]:03d}/{len(indices)-1:03d}\nEPE: {metrics["epe"]:.3f}'
        else:
            title = f'{metadata["sequence"]} - Frame {metadata["index"]:03d}\nEPE: {metrics["epe"]:.3f}'
        
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        return axes
    
    # Create animation
    anim = FuncAnimation(
        fig,
        update_frame,
        frames=len(indices),
        interval=1000//fps,
        blit=False
    )
    
    # Save animation
    if save_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
    else:
        anim.save(save_path, fps=fps, dpi=150)
    
    print(f"Saved animation to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize optical flow predictions from trained SNN model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='../checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../snn/configs/lightweight.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='../../blink_sim/output',
        help='Override data root path'
    )
    parser.add_argument(
        '--sample-idx',
        type=int,
        nargs='+',
        default=[0, 10, 20, 30, 40],
        help='Frame indices within the selected sequence to visualize (0-based)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../output/visualizations',
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--create-animation',
        action='store_true',
        help='Create animation of multiple samples'
    )
    parser.add_argument(
        '--animation-samples',
        type=int,
        default=20,
        help='Number of samples for animation (ignored if --sequence is specified)'
    )
    parser.add_argument(
        '--sequence',
        type=str,
        default="girl1_BaseballHit_0",
        help='Sequence name to visualize (e.g., girl1_BaseballHit_0)'
    )
    parser.add_argument(
        '--list-sequences',
        action='store_true',
        help='List all available sequences and exit'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots (only save)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on'
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SNN Optical Flow Visualization")
    print("="*60)
    
    # Load configuration
    print(f"\nLoading configuration from {args.config}")
    config = load_config(args.config)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, config, device)
    print(f"Model loaded successfully on {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(config, args.data_root)
    print(f"Loaded dataset with {len(dataset)} samples")

    # Map sequences to their dataset indices for convenient lookup
    sequences = get_sequences(dataset)
    
    # List sequences if requested
    if args.list_sequences:
        list_sequences(dataset)
        return

    # Validate requested sequence early
    if args.sequence and args.sequence not in sequences:
        print(f"\nError: Sequence '{args.sequence}' not found!")
        print("Available sequences:")
        for seq_name in sorted(sequences.keys()):
            print(f"  - {seq_name}")
        return
    
    # Visualize individual samples using the requested sequence
    if args.sequence:
        raw_sequence_indices = sequences[args.sequence]
        sequence_indices = raw_sequence_indices[:-1] if len(raw_sequence_indices) > 1 else raw_sequence_indices
        if len(raw_sequence_indices) != len(sequence_indices):
            print(f"\nNote: Skipping terminal frame for '{args.sequence}' to avoid missing forward-neighbor artifacts.")
        print(f"\nVisualizing sequence '{args.sequence}' with {len(sequence_indices)} usable frames (of {len(raw_sequence_indices)})")

        selected_pairs = []  # (frame_idx_in_sequence, dataset_idx)
        for frame_idx in args.sample_idx:
            if frame_idx >= len(sequence_indices):
                print(f"Warning: Frame {frame_idx} exceeds usable sequence length ({len(sequence_indices)}), skipping")
                continue
            selected_pairs.append((frame_idx, sequence_indices[frame_idx]))

        if not selected_pairs:
            print("No valid frame indices selected; nothing to visualize.")
            return
    else:
        # Fallback to global indices if no sequence is specified
        selected_pairs = [(None, idx) for idx in args.sample_idx]
        print(f"\nVisualizing samples (global indices): {args.sample_idx}")

    for frame_idx, dataset_idx in selected_pairs:
        if dataset_idx >= len(dataset):
            print(f"Warning: Sample {dataset_idx} exceeds dataset size, skipping")
            continue

        if frame_idx is not None:
            print(f"\nProcessing sequence frame {frame_idx} (dataset index {dataset_idx})...")
        else:
            print(f"\nProcessing sample {dataset_idx}...")

        sample = dataset[dataset_idx]
        
        # Run inference
        flow_pred, metrics = run_inference(model, sample, device)
        
        print(f"  EPE: {metrics['epe']:.4f}")
        print(f"  Max Flow GT: {metrics['max_flow_gt']:.2f}")
        print(f"  Max Flow Pred: {metrics['max_flow_pred']:.2f}")
        
        # Visualize
        save_path = output_dir / f"sample_{dataset_idx:04d}.png"
        visualize_flow_comparison(
            sample['input'],
            sample['flow'],
            flow_pred,
            metrics,
            save_path=str(save_path),
            show=not args.no_show
        )
    
    # Create animation if requested
    if args.create_animation:
        # Determine which samples to animate
        if args.sequence:
            # Use specific sequence
            if args.sequence not in sequences:
                print(f"\nError: Sequence '{args.sequence}' not found!")
                print("Available sequences:")
                for seq_name in sorted(sequences.keys()):
                    print(f"  - {seq_name}")
                return

            raw_animation_indices = sequences[args.sequence]
            animation_indices = raw_animation_indices#[:-1] if len(raw_animation_indices) > 1 else raw_animation_indices
            sequence_name = args.sequence
            if len(raw_animation_indices) != len(animation_indices):
                print(f"\nNote: Skipping terminal frame for '{args.sequence}' animation to avoid missing forward-neighbor artifacts.")
            print(f"\nCreating animation for sequence '{args.sequence}' with {len(animation_indices)} frames (of {len(raw_animation_indices)})...")
            animation_path = output_dir / f"{args.sequence}_animation.gif"
        else:
            # Select evenly spaced samples across all sequences
            total_samples = len(dataset)
            animation_indices = [
                int(i * total_samples / args.animation_samples)
                for i in range(args.animation_samples)
            ]
            sequence_name = None
            print(f"\nCreating animation with {args.animation_samples} samples...")
            animation_path = output_dir / "flow_animation.gif"
        
        create_flow_animation(
            model,
            dataset,
            animation_indices,
            device,
            str(animation_path),
            fps=5,
            sequence_name=sequence_name
        )
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
