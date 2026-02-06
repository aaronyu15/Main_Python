"""
Quick script to visualize model predictions on a single sample
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from snn.models import EventSNNFlowNetLite
from snn.dataset import OpticalFlowDataset
from snn.utils.visualization import visualize_flow
from utils import load_config


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    
    # Extract model config
    model_config = config.get('model', config)
    
    model = EventSNNFlowNetLite(config=model_config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation EPE: {checkpoint.get('best_val_epe', 'unknown'):.4f}")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Quick visualization of model predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to checkpoint file')
    parser.add_argument('--data-root', type=str, 
                      default='../../blink_sim/output/train_set',
                      help='Path to data')
    parser.add_argument('--sample-idx', type=int, default=0,
                      help='Sample index to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use')
    parser.add_argument('--save', type=str, default=None,
                      help='Path to save figure')
    parser.add_argument('--patch-mode', action='store_true',
                      help='Visualize patch extraction and show where patches come from')
    parser.add_argument('--show-full-image', action='store_true',
                      help='Show full image with patch locations marked')
    
    args = parser.parse_args()
    
    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model, config = load_checkpoint(args.checkpoint, device)
    
    # Create dataset config
    dataset_config = config.copy()
    dataset_config['data_root'] = args.data_root
    if args.patch_mode:
        dataset_config['patch_mode'] = True
        dataset_config['sparsify'] = True
        dataset_config['return_full_frame'] = True  # Get both patch and full frame
    
    # Load dataset
    dataset = OpticalFlowDataset(config=dataset_config)
    
    print(f"Dataset has {len(dataset)} samples")
    
    # Get sample
    sample_idx = np.random.randint(0, len(dataset)-1) if args.sample_idx == 0 else args.sample_idx
    sample = dataset[sample_idx]
    inputs = sample['input'].unsqueeze(0).to(device)  # Add batch dimension
    gt_flow = sample['flow']
    valid_mask = sample['valid_mask']
    
    # Get patch location from metadata if available
    patch_center = None
    if 'metadata' in sample and isinstance(sample['metadata'], dict):
        patch_center = sample['metadata'].get('center', None)
    
    flow_mag = torch.norm(gt_flow, p=2, dim=2)
    
    print(f"Input shape: {inputs.shape}")
    print(f"GT flow shape: {gt_flow.shape}")
    print(f"GT flow range: [{flow_mag.min():.2f}, {flow_mag.max():.2f}]")
    print(f"GT flow mean magnitude: {flow_mag.mean():.4f}")
    
    # If in patch mode, extract full frame data from sample
    if args.patch_mode or args.show_full_image:
        if 'full_input' in sample:
            # Data from same frame (patch mode with return_full_frame=True)
            full_input = sample['full_input']
            full_flow = sample['full_flow']
        else:
            # Fallback: load separately (non-patch mode)
            full_input = sample['input']
            full_flow = sample['flow']
        
        # Compute activity map
        activity_map = full_input.sum(dim=(0, 1)).numpy()  # [H, W]
        flow_mag_map = torch.norm(full_flow, p=2, dim=0).numpy()
        
        print(f"\nFull image stats:")
        print(f"  Activity range: [{activity_map.min():.2f}, {activity_map.max():.2f}]")
        print(f"  Flow magnitude range: [{flow_mag_map.min():.2f}, {flow_mag_map.max():.2f}]")
        print(f"  Flow magnitude mean: {flow_mag_map.mean():.4f}")
        print(f"  Pixels with flow > 0.1: {(flow_mag_map > 0.1).sum()}")
        print(f"  Pixels with activity > 5: {(activity_map > 5).sum()}")
    
    # Run inference
    with torch.no_grad():
        outputs = model(inputs)
        pred_flow = outputs['flow'].squeeze(0).cpu()  # Remove batch dimension
    
    print(f"Prediction shape: {pred_flow.shape}")
    
    # Compute metrics
    flow_error = torch.norm(pred_flow - gt_flow, p=2, dim=0)
    valid_error = flow_error * valid_mask.squeeze(0)
    epe = valid_error.sum() / (valid_mask.sum() + 1e-8)
    
    print(f"EPE: {epe.item():.4f}")
    
    # Find common max flow for consistent color scale
    gt_mag = torch.norm(gt_flow, p=2, dim=0).max().item()
    pred_mag = torch.norm(pred_flow, p=2, dim=0).max().item()
    max_flow = max(gt_mag, pred_mag)
    
    # Convert flows to Middlebury color using visualization utils
    gt_color = visualize_flow(gt_flow, max_flow) / 255.0  # Normalize to [0, 1]
    pred_color = visualize_flow(pred_flow, max_flow) / 255.0  # Normalize to [0, 1]
    error_map = flow_error.cpu().numpy()
    
    # Visualize
    if args.patch_mode or args.show_full_image:
        # Extended visualization showing full context - split into two windows
        
        # Window 1: Flow predictions and context
        fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Full image context
        full_flow_color = visualize_flow(full_flow, max_flow) / 255.0
        axes1[0, 0].imshow(full_flow_color)
        axes1[0, 0].set_title('Full Image - GT Flow (Middlebury)')
        axes1[0, 0].axis('off')
        
        im_act = axes1[0, 1].imshow(activity_map, cmap='hot')
        axes1[0, 1].set_title(f'Activity Map\nMax: {activity_map.max():.1f}')
        axes1[0, 1].axis('off')
        plt.colorbar(im_act, ax=axes1[0, 1], fraction=0.046, pad=0.04)
        
        im_flow_mag = axes1[0, 2].imshow(flow_mag_map, cmap='viridis')
        axes1[0, 2].set_title(f'Flow Magnitude\nMax: {flow_mag_map.max():.2f}')
        axes1[0, 2].axis('off')
        plt.colorbar(im_flow_mag, ax=axes1[0, 2], fraction=0.046, pad=0.04)
        
        # Draw patch location rectangles if we have patch center
        if patch_center is not None:
            from matplotlib.patches import Rectangle
            y_center, x_center = patch_center
            patch_size = dataset_config.get('patch_size', 128)
            half = patch_size // 2
            
            # Draw yellow rectangle on all three full-frame plots
            for ax in [axes1[0, 0], axes1[0, 1], axes1[0, 2]]:
                rect = Rectangle((x_center - half, y_center - half), 
                               patch_size, patch_size,
                               linewidth=2, edgecolor='yellow', facecolor='none')
                ax.add_patch(rect)
        
        # Row 2: Patch GT and prediction with Middlebury color scheme
        axes1[1, 0].imshow(gt_color)
        axes1[1, 0].set_title(f'Patch - GT Flow (Middlebury)\nMax: {gt_mag:.2f} px/frame')
        axes1[1, 0].axis('off')
        
        axes1[1, 1].imshow(pred_color)
        axes1[1, 1].set_title(f'Patch - Predicted Flow (Middlebury)\nMax: {pred_mag:.2f} px/frame')
        axes1[1, 1].axis('off')
        
        im_err = axes1[1, 2].imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max())
        axes1[1, 2].set_title(f'Patch - Error\nEPE: {epe.item():.4f}')
        axes1[1, 2].axis('off')
        plt.colorbar(im_err, ax=axes1[1, 2], fraction=0.046, pad=0.04)
        
        fig1.suptitle(f'Flow Predictions - Sample {args.sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Window 2: Event input analysis
        # Input shape: [1, num_bins, C, H, W] where C is polarities
        num_bins = inputs.shape[1]
        
        # Create a figure with num_bins + 2 subplots (sum, mask, and individual bins)
        ncols = min(num_bins + 2, 6)  # Limit columns to 6 for readability
        nrows = int(np.ceil((num_bins + 2) / ncols))
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        axes2 = axes2.flatten() if isinstance(axes2, np.ndarray) else [axes2]
        
        # Show summative event heatmap in first subplot
        patch_activity = inputs.squeeze(0).sum(dim=(0, 1)).cpu().numpy()  # Sum over bins and polarities
        im_patch_act = axes2[0].imshow(patch_activity, cmap='hot')
        axes2[0].set_title(f'Event Sum\nTotal: {patch_activity.sum():.1f}')
        axes2[0].axis('off')
        plt.colorbar(im_patch_act, ax=axes2[0], fraction=0.046, pad=0.04)
        
        # Show valid mask in second subplot
        axes2[1].imshow(valid_mask.squeeze(0).cpu().numpy(), cmap='gray')
        axes2[1].set_title('Valid Mask')
        axes2[1].axis('off')
        
        # Show individual time bins
        # For visualization, sum across polarity dimension for each bin
        event_bins = inputs.squeeze(0).sum(dim=1).cpu().numpy()  # [num_bins, H, W]
        
        for i in range(num_bins):
            ax_idx = i + 2  # Now starting at index 2
            if ax_idx < len(axes2):
                im_bin = axes2[ax_idx].imshow(event_bins[i], cmap='hot')
                axes2[ax_idx].set_title(f'Bin {i+1}/{num_bins}')
                axes2[ax_idx].axis('off')
                plt.colorbar(im_bin, ax=axes2[ax_idx], fraction=0.046, pad=0.04)
        
        # Hide any unused subplots
        for i in range(num_bins + 2, len(axes2)):
            axes2[i].axis('off')
        
        fig2.suptitle(f'Event Input Analysis - Sample {args.sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
    else:
        # Standard visualization for full images - split into two windows
        
        # Window 1: Flow predictions
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
        
        # Ground truth flow with Middlebury color scheme
        axes1[0, 0].imshow(gt_color)
        axes1[0, 0].set_title(f'GT Flow (Middlebury)\nMax: {gt_mag:.2f} px/frame')
        axes1[0, 0].axis('off')
        
        # Predicted flow with Middlebury color scheme
        axes1[0, 1].imshow(pred_color)
        axes1[0, 1].set_title(f'Predicted Flow (Middlebury)\nMax: {pred_mag:.2f} px/frame')
        axes1[0, 1].axis('off')
        
        # Error map
        im_err = axes1[1, 0].imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max())
        axes1[1, 0].set_title(f'Endpoint Error\nEPE: {epe.item():.4f}')
        axes1[1, 0].axis('off')
        plt.colorbar(im_err, ax=axes1[1, 0], fraction=0.046, pad=0.04)
        
        # Valid mask
        axes1[1, 1].imshow(valid_mask.squeeze(0).cpu().numpy(), cmap='gray')
        axes1[1, 1].set_title('Valid Mask')
        axes1[1, 1].axis('off')
        
        fig1.suptitle(f'Flow Predictions - Sample {args.sample_idx}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Window 2: Event input and statistics
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        
        # Event heatmap
        event_heatmap = inputs.squeeze(0).sum(dim=(0, 1)).cpu().numpy()
        im_events = axes2[0].imshow(event_heatmap, cmap='hot')
        axes2[0].set_title(f'Event Heatmap\nTotal Events: {event_heatmap.sum():.0f}')
        axes2[0].axis('off')
        plt.colorbar(im_events, ax=axes2[0], fraction=0.046, pad=0.04)
        
        # Flow distribution histogram
        flow_values = gt_flow[:, valid_mask.squeeze(0).bool()].cpu().numpy()
        if flow_values.shape[1] > 0:
            axes2[1].hist(np.linalg.norm(flow_values, axis=0), bins=50, alpha=0.7, label='GT')
            pred_values = pred_flow[:, valid_mask.squeeze(0).bool()].cpu().numpy()
            axes2[1].hist(np.linalg.norm(pred_values, axis=0), bins=50, alpha=0.7, label='Pred')
            axes2[1].set_title('Flow Magnitude Distribution')
            axes2[1].set_xlabel('Magnitude (px/frame)')
            axes2[1].set_ylabel('Count')
            axes2[1].legend()
        else:
            axes2[1].text(0.5, 0.5, 'No valid flow', ha='center', va='center')
            axes2[1].axis('off')
        
        fig2.suptitle(f'Event Input - Sample {args.sample_idx} - {sample["metadata"]}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    if args.save:
        # Save both figures if in multi-window mode
        if args.patch_mode or args.show_full_image:
            save_path1 = Path(args.save)
            save_path2 = save_path1.parent / f"{save_path1.stem}_events{save_path1.suffix}"
            fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
            fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path1} and {save_path2}")
        else:
            save_path1 = Path(args.save)
            save_path2 = save_path1.parent / f"{save_path1.stem}_events{save_path1.suffix}"
            fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
            fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path1} and {save_path2}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
