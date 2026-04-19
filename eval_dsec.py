"""
Evaluate trained SNN model on DSEC train/test set.

Computes EPE, angular error, outlier %, and percentage-error thresholds (1PE, 2PE, 3PE).

Usage:
    python eval_dsec.py \
        --checkpoint checkpoints/11_dsec_5bin_20ms_1/best_model.pth \
        --config snn/configs/event_snn_lite_dsec.yaml \
        --data-root ../dsec/train \
        --max-samples 50
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from comparisons.dsec.dsec_dataset import DSECOpticalFlowDataset
from snn.training import endpoint_error, calculate_outliers, angular_error
from utils import get_model, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DSEC dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML')
    parser.add_argument('--data-root', type=str, default='../dsec/train',
                        help='Path to DSEC data root (train or test)')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Specific sequence to evaluate (e.g., zurich_city_01_a)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--strict-load', action='store_true',
                        help='Strict checkpoint loading')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of samples')
    parser.add_argument('--integer-sim', action='store_true',
                        help='Use integer-only inference')
    return parser.parse_args()


def _extract_state_dict(checkpoint_obj):
    if 'model_state_dict' in checkpoint_obj:
        return checkpoint_obj['model_state_dict']
    if 'state_dict' in checkpoint_obj:
        return checkpoint_obj['state_dict']
    return checkpoint_obj


def build_eval_model(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if args.config is not None:
        config = load_config(args.config)
    else:
        config = checkpoint.get('config', None)

    if config is None:
        raise ValueError('No config found in checkpoint. Please provide --config.')

    model = get_model(config)
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict_load)

    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model = model.to(device)
    model.eval()
    model.disable_skip = True

    if args.integer_sim:
        if not config.get('quantized', False):
            raise ValueError('--integer-sim requires quantized config.')
        from snn.models.integer_inference import IntegerInferenceModel
        model = IntegerInferenceModel(model, config,
                                      accum_bit_width=config.get('accum_bit_width', 32))
        model = model.to(device)

    return model, config


def compute_percentage_errors(pred_flow: torch.Tensor, gt_flow: torch.Tensor, 
                              valid_mask: torch.Tensor, thresholds=[1.0, 2.0, 3.0]):
    """
    Compute bad-pixel percentage (nPE): percentage of valid pixels with EPE > threshold.
    
    Args:
        pred_flow: [B, 2, H, W]
        gt_flow: [B, 2, H, W]
        valid_mask: [B, 1, H, W]
        thresholds: list of EPE thresholds
    
    Returns:
        dict mapping threshold -> bad-pixel percentage
    """
    epe = torch.norm(pred_flow - gt_flow, dim=1, keepdim=True)  # [B, 1, H, W]
    epe_valid = epe[valid_mask > 0.5]
    
    result = {}
    for thresh in thresholds:
        pct = 100.0 * float((epe_valid > thresh).float().mean()) if epe_valid.numel() > 0 else 0.0
        result[f'{thresh:.0f}PE'] = pct
    return result


def main():
    args = parse_args()
    
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    print(f'Using device: {device}')
    
    model, config = build_eval_model(args, device)
    
    dataset_config = dict(config)
    dataset_config['data_root'] = args.data_root
    dataset_config['max_train_samples'] = args.max_samples
    
    dataset = DSECOpticalFlowDataset(config=dataset_config)
    if len(dataset) == 0:
        raise RuntimeError(f'No DSEC samples found in {args.data_root}')
    
    # Filter by sequence if specified
    if args.sequence is not None:
        filtered_samples = [s for s in dataset.samples if s['sequence'] == args.sequence]
        if not filtered_samples:
            raise RuntimeError(f'No samples found for sequence: {args.sequence}')
        dataset.samples = filtered_samples
        print(f'Filtered to sequence: {args.sequence} ({len(dataset.samples)} samples)')
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )
    
    print(f'Evaluating on {len(dataset)} samples')
    
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating DSEC'):
            inputs = batch['input'].to(device, non_blocking=True)
            gt_flow = batch['flow'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            metadata = batch['metadata']
            
            outputs = model(inputs)
            if isinstance(outputs, dict):
                pred_flow = outputs.get('flow', outputs.get('pred_flow', list(outputs.values())[0]))
            else:
                pred_flow = outputs
            
            # Skip if no valid pixels
            if valid_mask.sum() < 1:
                continue
            
            metrics = {}
            
            # Compute event activity mask: pixels where events occurred
            # inputs shape: [B, T, C, H, W]
            event_activity = inputs.sum(dim=(1, 2)) > 0  # [B, H, W]
            event_activity = event_activity.unsqueeze(1).float()  # [B, 1, H, W]
            event_mask = valid_mask.float() * event_activity  # combined mask
            
            # ---- Dense flow metrics (all valid pixels) ----
            epe_val = endpoint_error(pred_flow, gt_flow, valid_mask, proc='epe')
            metrics['epe_dense'] = float(epe_val) if isinstance(epe_val, torch.Tensor) else epe_val
            
            outliers = calculate_outliers(pred_flow, gt_flow, valid_mask, threshold=3.0)
            metrics['outliers_3px_dense'] = float(outliers) if isinstance(outliers, torch.Tensor) else outliers
            
            ae = angular_error(pred_flow, gt_flow, valid_mask, mode='gen')
            metrics['angular_error_dense'] = float(ae) if isinstance(ae, torch.Tensor) else ae
            
            pe = compute_percentage_errors(pred_flow, gt_flow, valid_mask)
            for k, v in pe.items():
                metrics[f'{k}_dense'] = v
            
            # ---- Masked flow metrics (only event regions) ----
            if event_mask.sum() > 0:
                epe_masked = endpoint_error(pred_flow, gt_flow, event_mask, proc='epe')
                metrics['epe_masked'] = float(epe_masked) if isinstance(epe_masked, torch.Tensor) else epe_masked
                
                outliers_masked = calculate_outliers(pred_flow, gt_flow, event_mask, threshold=3.0)
                metrics['outliers_3px_masked'] = float(outliers_masked) if isinstance(outliers_masked, torch.Tensor) else outliers_masked
                
                ae_masked = angular_error(pred_flow, gt_flow, event_mask, mode='gen')
                metrics['angular_error_masked'] = float(ae_masked) if isinstance(ae_masked, torch.Tensor) else ae_masked
                
                pe_masked = compute_percentage_errors(pred_flow, gt_flow, event_mask)
                for k, v in pe_masked.items():
                    metrics[f'{k}_masked'] = v
            else:
                # No event pixels
                metrics['epe_masked'] = np.nan
                metrics['outliers_3px_masked'] = np.nan
                metrics['angular_error_masked'] = np.nan
                for thresh in [1.0, 2.0, 3.0]:
                    metrics[f'{thresh:.0f}PE_masked'] = np.nan
            
            # Metadata
            metrics['sequence'] = str(metadata['sequence'][0])
            metrics['index'] = int(metadata['index'][0].item())
            
            all_metrics.append(metrics)
    
    # Aggregate
    if not all_metrics:
        print('No valid samples to evaluate.')
        return
    
    # Collect all unique metric keys
    all_keys = set()
    for m in all_metrics:
        all_keys.update(k for k in m.keys() if k not in ['sequence', 'index'])
    keys = sorted([k for k in all_keys if k not in ['sequence', 'index']])
    
    avg_metrics = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        # Filter out NaN values
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            avg_metrics[k] = np.mean(vals)
            avg_metrics[f'{k}_std'] = np.std(vals)
        else:
            avg_metrics[k] = np.nan
            avg_metrics[f'{k}_std'] = np.nan
    
    # Print results
    print('\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Data root: {args.data_root}')
    print(f'Samples: {len(all_metrics)}')
    print()
    
    print('DENSE FLOW (all valid pixels):')
    if 'epe_dense' in avg_metrics:
        print(f'  EPE:             {avg_metrics["epe_dense"]:.4f} ± {avg_metrics["epe_dense_std"]:.4f}')
        print(f'  Outliers (3px): {avg_metrics["outliers_3px_dense"]:.2f}% ± {avg_metrics["outliers_3px_dense_std"]:.2f}%')
        print(f'  Angular Error:   {avg_metrics["angular_error_dense"]:.2f}° ± {avg_metrics["angular_error_dense_std"]:.2f}°')
        print(f'  1PE (>1px):      {avg_metrics["1PE_dense"]:.2f}% ± {avg_metrics["1PE_dense_std"]:.2f}%')
        print(f'  2PE (>2px):      {avg_metrics["2PE_dense"]:.2f}% ± {avg_metrics["2PE_dense_std"]:.2f}%')
        print(f'  3PE (>3px):      {avg_metrics["3PE_dense"]:.2f}% ± {avg_metrics["3PE_dense_std"]:.2f}%')
    print()
    
    print('MASKED FLOW (only event regions):')
    if 'epe_masked' in avg_metrics:
        epe_m = avg_metrics['epe_masked']
        if not np.isnan(epe_m):
            print(f'  EPE:             {epe_m:.4f} ± {avg_metrics["epe_masked_std"]:.4f}')
            print(f'  Outliers (3px): {avg_metrics["outliers_3px_masked"]:.2f}% ± {avg_metrics["outliers_3px_masked_std"]:.2f}%')
            print(f'  Angular Error:   {avg_metrics["angular_error_masked"]:.2f}° ± {avg_metrics["angular_error_masked_std"]:.2f}°')
            print(f'  1PE (>1px):      {avg_metrics["1PE_masked"]:.2f}% ± {avg_metrics["1PE_masked_std"]:.2f}%')
            print(f'  2PE (>2px):      {avg_metrics["2PE_masked"]:.2f}% ± {avg_metrics["2PE_masked_std"]:.2f}%')
            print(f'  3PE (>3px):      {avg_metrics["3PE_masked"]:.2f}% ± {avg_metrics["3PE_masked_std"]:.2f}%')
        else:
            print('  (No event pixels in any sample)')
    print('='*60)
    
    # Save results
    results_path = Path('output') / f'eval_dsec_{Path(args.data_root).name}.txt'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        f.write('DSEC EVALUATION RESULTS\n')
        f.write('='*60 + '\n')
        f.write(f'Checkpoint: {args.checkpoint}\n')
        f.write(f'Data root: {args.data_root}\n')
        f.write(f'Samples: {len(all_metrics)}\n')
        f.write('\nDENSE FLOW (all valid pixels):\n')
        if 'epe_dense' in avg_metrics:
            f.write(f'  EPE:             {avg_metrics["epe_dense"]:.4f} ± {avg_metrics["epe_dense_std"]:.4f}\n')
            f.write(f'  Outliers (3px): {avg_metrics["outliers_3px_dense"]:.2f}% ± {avg_metrics["outliers_3px_dense_std"]:.2f}%\n')
            f.write(f'  Angular Error:   {avg_metrics["angular_error_dense"]:.2f}° ± {avg_metrics["angular_error_dense_std"]:.2f}°\n')
            f.write(f'  1PE (>1px):      {avg_metrics["1PE_dense"]:.2f}% ± {avg_metrics["1PE_dense_std"]:.2f}%\n')
            f.write(f'  2PE (>2px):      {avg_metrics["2PE_dense"]:.2f}% ± {avg_metrics["2PE_dense_std"]:.2f}%\n')
            f.write(f'  3PE (>3px):      {avg_metrics["3PE_dense"]:.2f}% ± {avg_metrics["3PE_dense_std"]:.2f}%\n')
        f.write('\nMASKED FLOW (only event regions):\n')
        if 'epe_masked' in avg_metrics:
            epe_m = avg_metrics['epe_masked']
            if not np.isnan(epe_m):
                f.write(f'  EPE:             {epe_m:.4f} ± {avg_metrics["epe_masked_std"]:.4f}\n')
                f.write(f'  Outliers (3px): {avg_metrics["outliers_3px_masked"]:.2f}% ± {avg_metrics["outliers_3px_masked_std"]:.2f}%\n')
                f.write(f'  Angular Error:   {avg_metrics["angular_error_masked"]:.2f}° ± {avg_metrics["angular_error_masked_std"]:.2f}°\n')
                f.write(f'  1PE (>1px):      {avg_metrics["1PE_masked"]:.2f}% ± {avg_metrics["1PE_masked_std"]:.2f}%\n')
                f.write(f'  2PE (>2px):      {avg_metrics["2PE_masked"]:.2f}% ± {avg_metrics["2PE_masked_std"]:.2f}%\n')
                f.write(f'  3PE (>3px):      {avg_metrics["3PE_masked"]:.2f}% ± {avg_metrics["3PE_masked_std"]:.2f}%\n')
            else:
                f.write('  (No event pixels in any sample)\n')
        f.write('='*60 + '\n')
        f.write('\nPer-sample results:\n')
        for m in all_metrics:
            f.write(f'{m["sequence"]}_{m["index"]:06d}:\n')
            f.write(f'  dense: EPE={m.get("epe_dense", np.nan):.4f}, AE={m.get("angular_error_dense", np.nan):.2f}°, ')
            f.write(f'1PE={m.get("1PE_dense", np.nan):.1f}%, 2PE={m.get("2PE_dense", np.nan):.1f}%, 3PE={m.get("3PE_dense", np.nan):.1f}%\n')
            f.write(f'  masked: EPE={m.get("epe_masked", np.nan):.4f}, AE={m.get("angular_error_masked", np.nan):.2f}°, ')
            f.write(f'1PE={m.get("1PE_masked", np.nan):.1f}%, 2PE={m.get("2PE_masked", np.nan):.1f}%, 3PE={m.get("3PE_masked", np.nan):.1f}%\n')
    
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
