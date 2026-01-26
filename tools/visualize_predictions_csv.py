"""
Visualization script for testing trained SNN optical flow models on real CSV event data

This script loads a trained model, runs inference on real CSV event data from blink_sim/output/real,
and creates visualizations showing the event stream and predicted optical flow.
Unlike visualize_predictions.py which uses HDF5 datasets, this works directly with CSV event files.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import yaml
import csv

import sys 
sys.path.insert(0, '..')
from snn.models import EventSNNFlowNetLite
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
            decay=config.get('decay', 2.0),
            threshold=config.get('threshold', 1.0),
            alpha=config.get('alpha', 10.0),
            quantize_weights=config.get('quantize_weights', False),
            quantize_activations=config.get('quantize_activations', False),
            quantize_mem=config.get('quantize_mem', False),
            weight_bit_width=config.get('weight_bit_width', 8),
            act_bit_width=config.get('act_bit_width', 8),
            output_bit_width=config.get('output_bit_width', 16),
            first_layer_bit_width=config.get('first_layer_bit_width', 8),
            mem_bit_width=config.get('mem_bit_width', 16),
            enable_logging=config.get('log_params', False),
            logger=None
        )

    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        val_epe = checkpoint.get('val_epe', None)
        if val_epe is not None:
            print(f"Validation EPE: {val_epe:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def load_events_from_csv(csv_path: str) -> np.ndarray:
    """
    Load events from CSV file
    
    CSV format: x,y,polarity,timestamp (one event per line)
    - x, y: pixel coordinates
    - polarity: 0 or 1
    - timestamp: timestamp in microseconds
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Events array [N, 4] where columns are [x, y, t, p]
        Polarity is converted to -1/+1 format
    """
    events_list = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x, y, pol, t = row
            # Convert polarity from 0/1 to -1/+1
            pol_converted = 1 if int(pol) == 1 else -1
            events_list.append([float(x), float(y), float(t), float(pol_converted)])
    
    return np.array(events_list, dtype=np.float32)


def events_to_voxel_grid(
    events: np.ndarray,
    num_bins: int,
    height: int = 320,
    width: int = 320
) -> torch.Tensor:
    """
    Convert event array to voxel grid representation
    
    Events format: [N, 4] where columns are [x, y, t, p]
    - x, y: pixel coordinates
    - t: timestamp
    - p: polarity (+1 or -1)
    
    Args:
        events: Event array [N, 4]
        num_bins: Number of temporal bins
        height: Image height
        width: Image width
    
    Returns:
        Voxel grid [num_bins, 2, H, W] for EventSNNFlowNetLite (polarity-separated)
    """
    if len(events) == 0:
        # Return zeros if no events
        return torch.zeros(num_bins, 2, height, width)
    
    # Parse events
    x = events[:, 0].astype(np.int32)
    y = events[:, 1].astype(np.int32)
    t = events[:, 2]
    p = events[:, 3]
    
    # Normalize timestamps to [0, num_bins)
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min) * (num_bins - 1e-6)
    else:
        t_norm = np.zeros_like(t)
    
    # Create voxel grid with separate polarity channels [num_bins, 2, H, W]
    # Channel 0 = positive events, Channel 1 = negative events
    voxel_grid = np.zeros((num_bins, 2, height, width), dtype=np.float32)
    
    # Distribute events into temporal bins with polarity separation
    for i in range(len(events)):
        bin_idx = int(t_norm[i])
        if 0 <= bin_idx < num_bins:
            if 0 <= x[i] < width and 0 <= y[i] < height:
                pol_idx = 0 if p[i] > 0 else 1  # Channel 0 = positive, 1 = negative
                voxel_grid[bin_idx, pol_idx, y[i], x[i]] += 1.0
    
    return torch.from_numpy(voxel_grid)


def create_time_windows(
    events: np.ndarray,
    window_duration_us: float,
    stride_us: float,
    start_time_us: Optional[float] = None,
    end_time_us: Optional[float] = None
) -> List[Tuple[int, int, float, float]]:
    """
    Create overlapping time windows for event processing
    
    Args:
        events: Event array [N, 4]
        window_duration_us: Duration of each window in microseconds
        stride_us: Time between window starts in microseconds
        start_time_us: Optional start time (default: first event timestamp)
        end_time_us: Optional end time (default: last event timestamp)
    
    Returns:
        List of (start_idx, end_idx, t_start, t_end) tuples
    """
    if len(events) == 0:
        return []
    
    timestamps = events[:, 2]
    
    if start_time_us is None:
        start_time_us = timestamps[0]
    if end_time_us is None:
        end_time_us = timestamps[-1]
    
    windows = []
    current_start = start_time_us
    
    while current_start + window_duration_us <= end_time_us:
        current_end = current_start + window_duration_us
        
        # Find events in this window using binary search
        start_idx = np.searchsorted(timestamps, current_start, side='left')
        end_idx = np.searchsorted(timestamps, current_end, side='left')
        
        windows.append((start_idx, end_idx, current_start, current_end))
        current_start += stride_us
    
    return windows


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Run inference on an input tensor
    
    Args:
        model: Trained model
        input_tensor: Input voxel grid [num_bins, 2, H, W]
        device: Device to run on
    
    Returns:
        Predicted flow [2, H, W]
    """
    # Add batch dimension and move to device
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Run inference
    output = model(input_batch)
    
    # Get predicted flow (remove batch dimension)
    if isinstance(output, dict):
        flow_pred = output['flow'].squeeze(0).cpu()
    else:
        flow_pred = output.squeeze(0).cpu()
    
    return flow_pred


def visualize_event_flow(
    input_events: torch.Tensor,
    flow_pred: torch.Tensor,
    save_path: Optional[str] = None,
    show: bool = True,
    frame_info: str = ""
):
    """
    Create visualization of events and predicted flow
    
    Args:
        input_events: Input event voxel grid [num_bins, 2, H, W]
        flow_pred: Predicted flow [2, H, W]
        save_path: Optional path to save figure
        show: Whether to display the figure
        frame_info: Additional information to display in title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert to numpy
    if isinstance(flow_pred, torch.Tensor):
        flow_pred = flow_pred.cpu().numpy()
    if isinstance(input_events, torch.Tensor):
        input_events = input_events.cpu().numpy()
    
    # Compute max flow for consistent visualization
    max_flow = max(np.sqrt((flow_pred**2).sum(axis=0)).max(), 1.0)
    
    # Input events - visualize by polarity (red=positive, blue=negative)
    # Sum across time bins: [num_bins, 2, H, W] -> [2, H, W]
    event_sum = input_events.sum(axis=0)  # [2, H, W]
    pos_events = event_sum[0]  # Positive events
    neg_events = event_sum[1]  # Negative events
    
    h, w = pos_events.shape
    event_rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # Positive events -> red channel
    if pos_events.max() > 0:
        event_rgb[:, :, 0] = pos_events / (pos_events.max() * 0.5)
    
    # Negative events -> blue channel
    if neg_events.max() > 0:
        event_rgb[:, :, 2] = neg_events / (neg_events.max() * 0.5)
    
    event_rgb = np.clip(event_rgb, 0, 1)
    axes[0].imshow(event_rgb)
    axes[0].set_title('Input Events (red=pos, blue=neg)')
    axes[0].axis('off')
    
    # Predicted flow (color-coded)
    flow_pred_color = flow_to_color(flow_pred, max_flow)
    axes[1].imshow(flow_pred_color)
    axes[1].set_title('Predicted Optical Flow')
    axes[1].axis('off')
    
    # Flow vectors (quiver plot)
    step = max(h // 20, 1)
    y, x = np.mgrid[0:h:step, 0:w:step]
    u_pred = flow_pred[0, ::step, ::step]
    v_pred = flow_pred[1, ::step, ::step]
    axes[2].quiver(x, y, u_pred, v_pred, scale=max_flow*5, color='cyan', alpha=0.7)
    axes[2].set_title('Flow Vectors')
    axes[2].set_xlim(0, w)
    axes[2].set_ylim(h, 0)
    axes[2].set_facecolor('black')
    axes[2].axis('off')
    
    plt.suptitle(
        f'Real Data Optical Flow Prediction\n{frame_info}',
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


def create_flow_animation_from_csv(
    model: torch.nn.Module,
    csv_path: str,
    device: torch.device,
    save_path: str,
    num_bins: int = 5,
    height: int = 320,
    width: int = 320,
    window_duration_ms: float = 50.0,
    stride_ms: float = 33.3,  # ~30 fps
    max_frames: Optional[int] = None,
    fps: int = 5
):
    """
    Create an animation showing predictions across time windows from CSV events
    
    Args:
        model: Trained model
        csv_path: Path to CSV file containing events
        device: Device to run inference on
        save_path: Path to save animation (should end in .gif)
        num_bins: Number of temporal bins for voxel grid
        height: Image height
        width: Image width
        window_duration_ms: Duration of each time window in milliseconds
        stride_ms: Time between window starts in milliseconds
        max_frames: Maximum number of frames to generate (None = all)
        fps: Frames per second for animation
    """
    print(f"\nLoading events from {csv_path}...")
    events = load_events_from_csv(csv_path)
    print(f"Loaded {len(events):,} events")
    
    if len(events) == 0:
        print("No events found in CSV file!")
        return
    
    # Convert milliseconds to microseconds
    window_duration_us = window_duration_ms * 1000.0
    stride_us = stride_ms * 1000.0
    
    # Create time windows
    print(f"Creating time windows (duration={window_duration_ms}ms, stride={stride_ms}ms)...")
    windows = create_time_windows(events, window_duration_us, stride_us)
    
    if max_frames is not None:
        windows = windows[:max_frames]
    
    print(f"Processing {len(windows)} time windows...")
    
    # Pre-process all frames
    frames_data = []
    for i, (start_idx, end_idx, t_start, t_end) in enumerate(windows):
        if i % 100 == 0:
            print(f"  Processing window {i+1}/{len(windows)}...")
        
        # Get events in this window
        window_events = events[start_idx:end_idx]
        
        # Convert to voxel grid
        voxel_grid = events_to_voxel_grid(window_events, num_bins, height, width)
        print(torch.min(voxel_grid), torch.max(voxel_grid), torch.mean(voxel_grid))
        
        # Run inference
        flow_pred = run_inference(model, voxel_grid, device)
        
        frames_data.append({
            'voxel_grid': voxel_grid.cpu().numpy(),
            'flow_pred': flow_pred.cpu().numpy(),
            'num_events': len(window_events),
            't_start_ms': t_start / 1000.0,
            't_end_ms': t_end / 1000.0
        })
    
    print("Creating animation...")
    
    # Create animation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    def update_frame(frame_idx):
        """Update function for animation"""
        frame_data = frames_data[frame_idx]
        
        voxel_grid = frame_data['voxel_grid']
        flow_pred = frame_data['flow_pred']
        
        # Compute max flow for consistent coloring
        max_flow = max(np.sqrt((flow_pred**2).sum(axis=0)).max(), 1.0)
        
        # Clear axes
        for ax in axes:
            ax.clear()
        
        # Input events
        event_sum = voxel_grid.sum(axis=0)  # [2, H, W]
        pos_events = event_sum[0]
        neg_events = event_sum[1]
        
        h, w = pos_events.shape
        event_rgb = np.zeros((h, w, 3), dtype=np.float32)
        
        if pos_events.max() > 0:
            event_rgb[:, :, 0] = pos_events / (pos_events.max() * 0.5)
        if neg_events.max() > 0:
            event_rgb[:, :, 2] = neg_events / (neg_events.max() * 0.5)
        
        event_rgb = np.clip(event_rgb, 0, 1)
        axes[0].imshow(event_rgb)
        axes[0].set_title(f'Events ({frame_data["num_events"]} events)')
        axes[0].axis('off')
        
        # Predicted flow
        flow_pred_color = flow_to_color(flow_pred, max_flow)
        axes[1].imshow(flow_pred_color)
        axes[1].set_title('Predicted Flow')
        axes[1].axis('off')
        
        # Flow vectors
        step = max(h // 20, 1)
        y, x = np.mgrid[0:h:step, 0:w:step]
        u_pred = flow_pred[0, ::step, ::step]
        v_pred = flow_pred[1, ::step, ::step]
        axes[2].quiver(x, y, u_pred, v_pred, scale=max_flow*5, color='cyan', alpha=0.7)
        axes[2].set_title('Flow Vectors')
        axes[2].set_xlim(0, w)
        axes[2].set_ylim(h, 0)
        axes[2].set_facecolor('black')
        axes[2].axis('off')
        
        # Update title
        title = (f'Frame {frame_idx+1}/{len(frames_data)} | '
                f'Time: {frame_data["t_start_ms"]:.1f}-{frame_data["t_end_ms"]:.1f}ms | '
                f'Max Flow: {max_flow:.2f} px/frame')
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        return axes
    
    # Create animation
    anim = FuncAnimation(
        fig,
        update_frame,
        frames=len(frames_data),
        interval=1000//fps,
        blit=False
    )
    
    # Save animation
    print(f"Saving animation to {save_path}...")
    if save_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
    else:
        anim.save(save_path, fps=fps, dpi=150)
    
    print(f"Saved animation to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize optical flow predictions from trained SNN model on real CSV event data'
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
        '--csv-file',
        type=str,
        required=True,
        help='Path to CSV file containing events'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../output/visualizations_csv',
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--window-duration',
        type=float,
        default=33.33,
        help='Duration of each time window in milliseconds (default: 50ms)'
    )
    parser.add_argument(
        '--stride',
        type=float,
        default=33.33,
        help='Time between window starts in milliseconds (default: 33.3ms = ~30fps)'
    )
    parser.add_argument(
        '--num-bins',
        type=int,
        default=5,
        help='Number of temporal bins for voxel grid (default: 5)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=320,
        help='Image height (default: 320)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=320,
        help='Image width (default: 320)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to generate (default: all)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for output animation (default: 5)'
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
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    print("="*60)
    print("SNN Optical Flow Visualization - Real CSV Data")
    print("="*60)
    
    # Load configuration
    print(f"\nLoading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override num_bins from config if specified
    num_bins = config.get('num_bins', config.get('in_channels', args.num_bins))
    print(f"Using {num_bins} temporal bins")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, config, device)
    print(f"Model loaded successfully on {device}")
    
    # Create animation
    output_filename = csv_path.stem + '_flow_animation.gif'
    output_path = output_dir / output_filename
    
    create_flow_animation_from_csv(
        model=model,
        csv_path=str(csv_path),
        device=device,
        save_path=str(output_path),
        num_bins=num_bins,
        height=args.height,
        width=args.width,
        window_duration_ms=args.window_duration,
        stride_ms=args.stride,
        max_frames=args.max_frames,
        fps=args.fps
    )
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
