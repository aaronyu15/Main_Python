import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import csv
import sys
from visualize_predictions import visualize_events, flow_to_color

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_config, build_model
from snn.utils import flow_to_color, visualize_events


def load_events_from_csv(csv_path: str, use_polarity: bool) -> np.ndarray:
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
            if use_polarity:
                pol_converted = 1 if int(pol) == 1 else -1
            else:
                pol_converted = 1  # Default polarity if not used
            events_list.append([float(x), float(y), float(t), float(pol_converted)])
    
    return np.array(events_list, dtype=np.float32)


def events_to_voxel_grid(
    events: np.ndarray,
    num_bins: int,
    height: int = 320,
    width: int = 320,
    use_polarity: bool = False
) -> torch.Tensor:

    if len(events) == 0:
        if use_polarity:
            return torch.zeros(num_bins, 2, 320, 320) 
        else:
            return torch.zeros(num_bins, 1, 320, 320)  
        
    x = events[:, 0].astype(np.int32)
    y = events[:, 1].astype(np.int32)
    t = events[:, 2]
    p = events[:, 3]
        
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min) * (num_bins - 1e-6)
    else:
        t_norm = np.zeros_like(t)
        
    if use_polarity:
        voxel_grid = np.zeros((num_bins, 2, height, width), dtype=np.float32)
    else:
        voxel_grid = np.zeros((num_bins, 1, height, width), dtype=np.float32)
        
    for i in range(len(events)):
        bin_idx = int(t_norm[i])
        if 0 <= bin_idx < num_bins:
            if 0 <= x[i] < width and 0 <= y[i] < height:
                if use_polarity:
                    pol_idx = 0 if p[i] > 0 else 1  
                else:
                    pol_idx = 0 
                voxel_grid[bin_idx, pol_idx, y[i], x[i]] += 1.0
        
    return torch.from_numpy(voxel_grid)


def create_time_windows(
    events: np.ndarray,
    window_duration_us: float,
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
        current_start += window_duration_us
    
    return windows


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    output = model(input_batch)
    
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    flow_pred = flow_pred.cpu().numpy()
    input_events = input_events.cpu().numpy()
    
    max_flow = max(np.sqrt((flow_pred**2).sum(axis=0)).max(), 1.0)
    
    event_rgb = visualize_events(input_events)
    h, w = event_rgb.shape[:2]
    axes[0].imshow(event_rgb)
    axes[0].set_title('Input Events')
    axes[0].axis('off')
    
    flow_pred_color = flow_to_color(flow_pred, max_flow)
    axes[1].imshow(flow_pred_color)
    axes[1].set_title('Predicted Optical Flow')
    axes[1].axis('off')
    
    step = max(h // 20, 1)
    y, x = np.mgrid[0:h:step, 0:w:step]
    u_pred = flow_pred[0, ::step, ::step]
    v_pred = -flow_pred[1, ::step, ::step]  # Negate v for image coordinate system
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
    use_polarity: bool = False,
    height: int = 320,
    width: int = 320,
    window_duration_us: float = 50.0,
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
        window_duration_us: Duration of each time window in milliseconds
        stride_ms: Time between window starts in milliseconds
        fps: Frames per second for animation
    """
    print(f"\nLoading events from {csv_path}...")
    events = load_events_from_csv(csv_path, use_polarity=use_polarity)
    print(f"Loaded {len(events):,} events")
    
    if len(events) == 0:
        print("No events found in CSV file!")
        return
    
    
    # Create time windows
    windows = create_time_windows(events, window_duration_us, window_duration_us)
    
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
        event_rgb = visualize_events(voxel_grid)
        h, w = event_rgb.shape[:2]
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
        v_pred = -flow_pred[1, ::step, ::step]  # Negate v for image coordinate system
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
        '--num-bins',
        type=int,
        default=5,
        help='Number of temporal bins for voxel grid (default: 5)'
    )

    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    
    print(f"\nLoading model from {args.checkpoint}")
    model, config = build_model(None, device=str(device), train=False, checkpoint_path=args.checkpoint)

    num_bins = config.get('num_bins', 5)
    bin_interval_us = config.get('bin_interval_us', 5000.0)  # ms per bin

    window_duration_us = num_bins * bin_interval_us
    use_polarity = config.get('use_polarity', False)
    
    # Create animation
    output_filename = csv_path.stem + '_flow_animation.gif'
    output_path = output_dir / output_filename
    
    create_flow_animation_from_csv(
        model=model,
        csv_path=str(csv_path),
        device=device,
        save_path=str(output_path),
        num_bins=num_bins,
        use_polarity=use_polarity,
        height=320,
        width=320,
        window_duration_us=window_duration_us,
        fps=30
    )
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
