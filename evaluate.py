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

from snn.models import EventSNNFlowNetLite
from snn.dataset import OpticalFlowDataset
from snn.training import endpoint_error, calculate_outliers, angular_error, epe_weighted_angular_error

from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SNN Optical Flow Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='../blink_sim/output/test_set',
                      help='Root directory for dataset')
    parser.add_argument('--output-dir', type=str, default='./results',
                      help='Directory to save results')
    parser.add_argument('--num-samples', type=int, default=None,
                      help='Number of samples to evaluate (None = all)')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                      help='Directory for logs')
    
    return parser.parse_args()


def evaluate(args):
    """Main evaluation function"""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    

    # Load model
    model, config = build_model(None, device, train=False, checkpoint_path=args.checkpoint)
    
    # Build dataset
    dataset_config = config.copy()
    dataset_config['data_root'] = args.data_root
    dataset_config['max_train_samples'] = args.num_samples
    
    dataset = OpticalFlowDataset(config=dataset_config)
    
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
            metrics = {}
            metrics['epe'] = endpoint_error(pred_flow, gt_flow, valid_mask)
            metrics['outliers'] = calculate_outliers(pred_flow, gt_flow, valid_mask, threshold=3.0)
            metrics['angular_error'] = angular_error(pred_flow, gt_flow, valid_mask)
            metrics['epe_weighted_angular_error'] = epe_weighted_angular_error(pred_flow, gt_flow, inputs, valid_mask)
            metrics['valid_pixels'] = valid_mask.sum().item()

            activity_patch = inputs.sum(dim=(1,2))
            low_activity_mask = (activity_patch < 1)
            low_activity_mask = low_activity_mask.unsqueeze(1)

            valid_mask[low_activity_mask] = 0.0
            metrics['epe_mask'] = endpoint_error(pred_flow, gt_flow, valid_mask)
            metrics['outliers_mask'] = calculate_outliers(pred_flow, gt_flow, valid_mask, threshold=3.0)
            metrics['angular_error_mask'] = angular_error(pred_flow, gt_flow, valid_mask)
            metrics['epe_weighted_angular_error_mask'] = epe_weighted_angular_error(pred_flow, gt_flow, inputs, valid_mask)
            metrics['valid_pixels_mask'] = valid_mask.sum().item()

            metrics['sequence'] = metadata['sequence'][0]
            metrics['index'] = metadata['index'][0].item()
            
            all_metrics.append(metrics)
            
    
    # Compute average metrics
    avg_metrics = {}
    for key in ['epe', 'outliers', 'angular_error', 'epe_mask', 'outliers_mask', 'angular_error_mask', 'epe_weighted_angular_error', 'epe_weighted_angular_error_mask']:
        values = [m[key].cpu().numpy() if type(m[key]) == torch.Tensor else m[key] for m in all_metrics]
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
    print(f"  EPE Weighted Angular Error: {avg_metrics['epe_weighted_angular_error']:.2f}° ± {avg_metrics['epe_weighted_angular_error_std']:.2f}°")
    print(f"  EPE (Masked): {avg_metrics['epe_mask']:.4f} ± {avg_metrics['epe_mask_std']:.4f}")
    print(f"  Outliers (Masked): {avg_metrics['outliers_mask']:.2f}% ± {avg_metrics['outliers_mask_std']:.2f}%")
    print(f"  Angular Error (Masked): {avg_metrics['angular_error_mask']:.2f}° ± {avg_metrics['angular_error_mask_std']:.2f}°")
    print(f"  EPE Weighted Angular Error (Masked): {avg_metrics['epe_weighted_angular_error_mask']:.2f}° ± {avg_metrics['epe_weighted_angular_error_mask_std']:.2f}°")
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
        f.write(f"  EPE Weighted Angular Error: {avg_metrics['epe_weighted_angular_error']:.2f}° ± {avg_metrics['epe_weighted_angular_error_std']:.2f}°\n")
        f.write(f"  EPE (Masked): {avg_metrics['epe_mask']:.4f} ± {avg_metrics['epe_mask_std']:.4f}\n")
        f.write(f"  Outliers (Masked): {avg_metrics['outliers_mask']:.2f}% ± {avg_metrics['outliers_mask_std']:.2f}%\n")
        f.write(f"  Angular Error (Masked): {avg_metrics['angular_error_mask']:.2f}° ± {avg_metrics['angular_error_mask_std']:.2f}°\n")
        f.write(f"  EPE Weighted Angular Error (Masked): {avg_metrics['epe_weighted_angular_error_mask']:.2f}° ± {avg_metrics['epe_weighted_angular_error_mask_std']:.2f}°\n")
        f.write("\n" + "="*50 + "\n")
        f.write("\nPer-sample results:\n")
        for m in all_metrics:
            f.write(f"{m['sequence']}_{m['index']:06d}: \n")
            f.write(f"EPE={m['epe']:.4f}, Outliers={m['outliers']:.2f}%, AngErr={m['angular_error']:.2f}°, EPE Weighted AngErr={m['epe_weighted_angular_error']:.2f}°, valid_pixels={m['valid_pixels']}\n")
            f.write(f"EPE masked={m['epe_mask']:.4f}, Outliers masked={m['outliers_mask']:.2f}%, AngErr masked={m['angular_error_mask']:.2f}°, EPE Weighted AngErr masked={m['epe_weighted_angular_error_mask']:.2f}°, valid_pixels_mask={m['valid_pixels_mask']}\n")
    
    print(f"\nResults saved to {results_file}")
    



if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
