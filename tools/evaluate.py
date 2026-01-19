"""
Evaluation script for trained SNN models
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys 
sys.path.insert(0, '..')
from snn.models import EventSNNFlowNetLite
from snn.data import OpticalFlowDataset
from snn.utils import compute_metrics, visualize_flow, plot_flow_comparison, save_flow_image


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SNN Optical Flow Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='../blink_sim/output',
                      help='Root directory for dataset')
    parser.add_argument('--output-dir', type=str, default='./results',
                      help='Directory to save results')
    parser.add_argument('--save-visualizations', action='store_true',
                      help='Save flow visualizations')
    parser.add_argument('--num-samples', type=int, default=None,
                      help='Number of samples to evaluate (None = all)')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    config = checkpoint.get('config', {})
    
    # Build model
    model_type = config.get('model_type', 'SpikingFlowNet')

    if model_type == 'EventSNNFlowNetLite':
        model = EventSNNFlowNetLite(
            base_ch=config.get('base_ch', 32),
            tau=config.get('tau', 2.0),
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Best validation EPE: {checkpoint.get('best_val_epe', 'unknown')}")
    
    return model, config


def evaluate(args):
    """Main evaluation function"""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Build dataset
    dataset = OpticalFlowDataset(
        data_root=args.data_root,
        split=None,
        transform=None,
        use_events=config.get('use_events', True),
        num_bins=config.get('in_channels', 5),
        crop_size=config.get('crop_size', (320, 320)),
        max_samples=args.num_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Evaluating on {len(dataset)} samples")
    
    # Metrics accumulator
    all_metrics = []
    
    # Evaluate
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move to device
            inputs = batch['input'].to(device)
            gt_flow = batch['flow'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            metadata = batch['metadata']
            
            # Forward pass
            outputs = model(inputs)
            pred_flow = outputs['flow']
            
            # Compute metrics
            metrics = compute_metrics(pred_flow, gt_flow, valid_mask)
            metrics['sequence'] = metadata['sequence'][0]
            metrics['index'] = metadata['index'][0].item()
            
            all_metrics.append(metrics)
            
            # Save visualizations
            if args.save_visualizations:
                vis_dir = output_dir / 'visualizations'
                vis_dir.mkdir(exist_ok=True)
                
                seq_name = metadata['sequence'][0]
                frame_idx = metadata['index'][0].item()
                
                # Save flow visualization
                save_path = vis_dir / f'{seq_name}_{frame_idx:06d}_pred.png'
                save_flow_image(pred_flow[0], str(save_path))
                
                # Save comparison plot
                comparison_path = vis_dir / f'{seq_name}_{frame_idx:06d}_comparison.png'
                plot_flow_comparison(
                    pred_flow[0],
                    gt_flow[0],
                    inputs[0],
                    str(comparison_path)
                )
    
    # Compute average metrics
    avg_metrics = {}
    for key in ['epe', 'outliers', 'angular_error']:
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of samples: {len(all_metrics)}")
    print(f"\nAverage Metrics:")
    print(f"  EPE: {avg_metrics['epe']:.4f} ± {avg_metrics['epe_std']:.4f}")
    print(f"  Outliers: {avg_metrics['outliers']:.2f}% ± {avg_metrics['outliers_std']:.2f}%")
    print(f"  Angular Error: {avg_metrics['angular_error']:.2f}° ± {avg_metrics['angular_error_std']:.2f}°")
    print("="*50)
    
    # Save results to file
    results_file = output_dir / f'results.txt'
    with open(results_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Number of samples: {len(all_metrics)}\n")
        f.write(f"\nAverage Metrics:\n")
        f.write(f"  EPE: {avg_metrics['epe']:.4f} ± {avg_metrics['epe_std']:.4f}\n")
        f.write(f"  Outliers: {avg_metrics['outliers']:.2f}% ± {avg_metrics['outliers_std']:.2f}%\n")
        f.write(f"  Angular Error: {avg_metrics['angular_error']:.2f}° ± {avg_metrics['angular_error_std']:.2f}°\n")
        f.write("\n" + "="*50 + "\n")
        f.write("\nPer-sample results:\n")
        for m in all_metrics:
            f.write(f"{m['sequence']}_{m['index']:06d}: EPE={m['epe']:.4f}, "
                   f"Outliers={m['outliers']:.2f}%, AngErr={m['angular_error']:.2f}°\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Save metrics as numpy array
    metrics_file = output_dir / f'metrics.npy'
    np.save(metrics_file, all_metrics)
    print(f"Metrics saved to {metrics_file}")


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
