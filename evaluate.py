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
    model, config = build_model(None, device, train=False, checkpoint_path=args.checkpoint, strict=True)
    model.disable_skip = True
    
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

            # Preserve original mask for unmasked metrics before any downstream edits
            valid_mask_full = valid_mask.clone()

            # Forward pass
            outputs = model(inputs)
            pred_flow = outputs['flow']
            
            # Compute metrics
            metrics = {}
            metrics['epe'] = endpoint_error(pred_flow, gt_flow, valid_mask_full, proc="epe")
            metrics['outliers'] = calculate_outliers(pred_flow, gt_flow, valid_mask_full, threshold=3.0)
            metrics['angular_error'] = angular_error(pred_flow, gt_flow, valid_mask_full, mode="gen")
            metrics['epe_weighted_angular_error'] = epe_weighted_angular_error(pred_flow, gt_flow, inputs, valid_mask, proc="epe")
            metrics['valid_pixels'] = valid_mask.sum().item()

            activity_patch = inputs.sum(dim=(1,2))
            low_activity_mask = (activity_patch < 1)
            low_activity_mask = low_activity_mask.unsqueeze(1)

            valid_mask[low_activity_mask] = 0.0
            metrics['epe_mask'] = endpoint_error(pred_flow, gt_flow, valid_mask, proc="epe")
            metrics['outliers_mask'] = calculate_outliers(pred_flow, gt_flow, valid_mask, threshold=3.0)
            metrics['angular_error_mask'] = angular_error(pred_flow, gt_flow, valid_mask, mode="gen")
            metrics['epe_weighted_angular_error_mask'] = epe_weighted_angular_error(pred_flow, gt_flow, inputs, valid_mask, proc="epe")
            metrics['valid_pixels_mask'] = valid_mask.sum().item()

            dir_epe_mask = endpoint_error(pred_flow, gt_flow, valid_mask, proc="epe", mode="directional")
            dir_ang_mask = angular_error(pred_flow, gt_flow, valid_mask, mode="directional")
            for name, val in dir_epe_mask.items():
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().item()
                metrics[f'epe_mask_dir_{name}'] = float(val)
            for name, val in dir_ang_mask.items():
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().item()
                metrics[f'ang_mask_dir_{name}'] = float(val)

            metrics['sequence'] = metadata['sequence'][0]
            metrics['index'] = metadata['index'][0].item()
            
            all_metrics.append(metrics)
            
    
    # Compute average metrics
    avg_metrics = {}
    numeric_keys = [k for k in all_metrics[0].keys() if k not in ['sequence', 'index']]
    for key in numeric_keys:
        values = []
        for m in all_metrics:
            v = m[key]
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            values.append(v)
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
    def fmt_dir(prefix: str, decimals: int = 4):
        keys = ['left', 'right', 'up', 'down']
        return ', '.join([f"{k}: {avg_metrics[f'{prefix}{k}']:.{decimals}f}" for k in keys])

    print(f"  EPE Weighted Angular Error (Masked): {avg_metrics['epe_weighted_angular_error_mask']:.2f}° ± {avg_metrics['epe_weighted_angular_error_mask_std']:.2f}°")
    print(f"  Directional EPE (masked): {fmt_dir('epe_mask_dir_')}")
    print(f"  Directional Angular Error (masked): {fmt_dir('ang_mask_dir_', decimals=2)}")
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
        f.write(f"  EPE Weighted Angular Error: {avg_metrics['epe_weighted_angular_error']:.2f} ± {avg_metrics['epe_weighted_angular_error_std']:.2f}\n")

        f.write(f"\n")
        f.write(f"  EPE (Masked): {avg_metrics['epe_mask']:.4f} ± {avg_metrics['epe_mask_std']:.4f}\n")
        f.write(f"  Outliers (Masked): {avg_metrics['outliers_mask']:.2f}% ± {avg_metrics['outliers_mask_std']:.2f}%\n")
        f.write(f"  Angular Error (Masked): {avg_metrics['angular_error_mask']:.2f}° ± {avg_metrics['angular_error_mask_std']:.2f}°\n")
        f.write(f"  EPE Weighted Angular Error (Masked): {avg_metrics['epe_weighted_angular_error_mask']:.2f} ± {avg_metrics['epe_weighted_angular_error_mask_std']:.2f}\n")
        f.write(f"  Directional EPE (masked): {fmt_dir('epe_mask_dir_')}\n")
        f.write(f"  Directional Angular Error (masked): {fmt_dir('ang_mask_dir_', decimals=2)}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("\nPer-sample results:\n")
        for m in all_metrics:
            f.write(f"{m['sequence']}_{m['index']:06d}: \n")
            f.write(f"EPE masked={m['epe_mask']:.4f}, Outliers masked={m['outliers_mask']:.2f}%, AngErr masked={m['angular_error_mask']:.2f}°, EPE Weighted AngErr masked={m['epe_weighted_angular_error_mask']:.2f}°, valid_pixels_mask={m['valid_pixels_mask']}\n")
            f.write(f"  Directional EPE masked: {{'left': {m['epe_mask_dir_left']:.4f}, 'right': {m['epe_mask_dir_right']:.4f}, 'up': {m['epe_mask_dir_up']:.4f}, 'down': {m['epe_mask_dir_down']:.4f}}}\n")
            f.write(f"  Directional AngErr masked: {{'left': {m['ang_mask_dir_left']:.2f}, 'right': {m['ang_mask_dir_right']:.2f}, 'up': {m['ang_mask_dir_up']:.2f}, 'down': {m['ang_mask_dir_down']:.2f}}}\n")
    
    print(f"\nResults saved to {results_file}")
    



if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
