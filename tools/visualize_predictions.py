import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import sys

# Add parent directory to path to import from utils and snn
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_config, build_model
from snn.dataset import OpticalFlowDataset
from snn.utils import flow_to_color, visualize_events


def get_sequences(dataset: OpticalFlowDataset) -> Dict[str, list]:
    sequences = {}
    for idx, sample_info in enumerate(dataset.samples):
        seq_name = sample_info['sequence']
        if seq_name not in sequences:
            sequences[seq_name] = []
        sequences[seq_name].append(idx)
    
    return sequences


def list_sequences(dataset: OpticalFlowDataset):
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
    input_tensor = sample['input'].unsqueeze(0).to(device)
    flow_gt = sample['flow']
    
    output = model(input_tensor)
    
    flow_pred = output['flow'].squeeze(0).cpu()
    
    epe = torch.norm(flow_pred - flow_gt, p=2, dim=0).mean()
    masked_epe = torch.norm((flow_pred - flow_gt) * (input_tensor.cpu().sum(dim=1).squeeze(0) > 0), p=2, dim=0).mean()
    
    metrics = {
        'epe': epe.item(),
        'masked_epe': masked_epe.item(),
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
    
    flow_gt = flow_gt.cpu().numpy()
    flow_pred = flow_pred.cpu().numpy()
    input_events = input_events.cpu().numpy()
    
    max_flow = min(
        np.sqrt((flow_gt**2).sum(axis=0)).max(),
        np.sqrt((flow_pred**2).sum(axis=0)).max(),
        1.0
    )
    
    # Row 1: Input visualization

    # With polarity
    if input_events.shape[1] == 2:
        event_sum = input_events.sum(axis=0)  # [2, H, W]
        pos_events = event_sum[0]  # Positive events
        neg_events = event_sum[1]  # Negative events
    else:  
    # no polarity
        event_sum = input_events.sum(axis=0)  # [2, H, W]
        pos_events = event_sum[0]  # Positive events
        neg_events = np.zeros_like(pos_events)  # No negative events
    
    h, w = pos_events.shape
    event_rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    if pos_events.max() > 0:
        event_rgb[:, :, 0] = pos_events / pos_events.max()
    
    if neg_events.max() > 0:
        event_rgb[:, :, 2] = neg_events / neg_events.max()
    
    event_rgb = np.clip(event_rgb, 0, 1)
    axes[0, 0].imshow(event_rgb)
    axes[0, 0].set_title('Input Events')
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
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
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
            for a in ax:
                a.clear()
        
        # Input events - visualize by polarity (red=positive, blue=negative)
        event_rgb = visualize_events(input_events)
        axes[0, 0].imshow(event_rgb)
        axes[0, 0].set_title(f'Events (Frame {metadata["index"]:03d})')
        axes[0, 0].axis('off')
        
        # Ground truth
        flow_gt_color = flow_to_color(flow_gt, max_flow)
        axes[0, 1].imshow(flow_gt_color)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Prediction
        flow_pred_color = flow_to_color(flow_pred, max_flow)
        axes[0, 2].imshow(flow_pred_color)
        axes[0, 2].set_title(f'Prediction (EPE: {metrics["epe"]:.3f})')
        axes[0, 2].axis('off')

        # Masked by input events
        
        # Ground truth
        input_mask = input_events.sum(axis=0) > 0  

        flow_gt_masked = flow_gt * input_mask
        flow_gt_color = flow_to_color(flow_gt_masked, max_flow)
        axes[1, 1].imshow(flow_gt_color)
        axes[1, 1].set_title('Ground Truth')
        axes[1, 1].axis('off')
        
        # Prediction
        flow_pred_masked = flow_pred * input_mask
        flow_pred_color = flow_to_color(flow_pred_masked, max_flow)
        axes[1, 2].imshow(flow_pred_color)
        axes[1, 2].set_title(f'Prediction (EPE: {metrics["masked_epe"]:.3f})')
        axes[1, 2].axis('off')
        
        # Update title
        if sequence_name:
            title = f'{sequence_name} - Frame {metadata["index"]:03d}/{len(indices)-1:03d}\nEPE: {metrics["epe"]:.3f}, Masked EPE: {metrics["masked_epe"]:.3f}'
        else:
            title = f'{metadata["sequence"]} - Frame {metadata["index"]:03d}\nEPE: {metrics["epe"]:.3f}, Masked EPE: {metrics["masked_epe"]:.3f}'
        
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
        default='../../blink_sim/output/train',
        help='Override data root path'
    )
    parser.add_argument(
        '--sample-idx',
        type=int,
        nargs='+',
        default=[],
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
        type=bool,
        default=True,
        help='Create animation of multiple samples'
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
    model, _ = build_model(config, device=str(device), train=False, checkpoint_path=args.checkpoint)
    print(f"Model loaded successfully on {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_config = config.copy()
    if args.data_root is not None:
        dataset_config['data_root'] = args.data_root
    dataset = OpticalFlowDataset(config=dataset_config)

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
        list_sequences(dataset)
        return
    
    sequence_indices = sequences[args.sequence]
    print(f"\nVisualizing sequence '{args.sequence}' with {len(sequence_indices)} usable frames (of {len(sequence_indices)})")

    selected_pairs = []
    if len(args.sample_idx) > 0:
        for frame_idx in args.sample_idx:
            if frame_idx >= len(sequence_indices):
                print(f"Warning: Frame {frame_idx} exceeds usable sequence length ({len(sequence_indices)}), skipping")
                continue
            selected_pairs.append((frame_idx, sequence_indices[frame_idx]))

        for frame_idx, dataset_idx in selected_pairs:
            if frame_idx is not None:
                print(f"\nProcessing sequence frame {frame_idx} (dataset index {dataset_idx})...")

            sample = dataset[dataset_idx]

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
    
    if args.create_animation:

        animation_indices = sequences[args.sequence]
        sequence_name = args.sequence
        animation_path = output_dir / f"{args.sequence}_animation.gif"
        
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
