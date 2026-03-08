"""
Evaluation script for trained SNN models.

Supports both full-precision and quantized (PTQ/QAT) models.
Logs TensorBoard visualizations (events, flow, masks) alongside numeric metrics.

Usage:
    # Evaluate full-precision model (config embedded in checkpoint)
    python evaluate.py --checkpoint checkpoints/teacher_10000u/best_model.pth

    # Evaluate quantized model (requires --config for quant settings)
    python evaluate.py \
        --checkpoint checkpoints/ptq_8bit/ptq_model.pth \
        --config snn/configs/event_snn_lite_8bit.yaml \
        --quantized

    # Custom data root and TensorBoard log dir
    python evaluate.py \
        --checkpoint checkpoints/teacher_10000u/best_model.pth \
        --data-root ../blink_sim/output/test_set \
        --log-dir logs/eval_teacher
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
from snn.utils.logger import Logger
from snn.utils.visualization import visualize_flow
from torchvision.utils import make_grid

from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SNN Optical Flow Model')

    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to config YAML (required for quantized models, optional for full-precision)')
    parser.add_argument('--quantized', action='store_true',
                      help='Load as a quantized model (applies quant config and sets PTQ mode)')
    parser.add_argument('--data-root', type=str, default=None,
                      help='Root directory for dataset (overrides config)')
    parser.add_argument('--output-dir', type=str, default='./results',
                      help='Directory to save text results')
    parser.add_argument('--log-dir', type=str, default='./logs/eval',
                      help='Directory for TensorBoard logs')
    parser.add_argument('--num-samples', type=int, default=None,
                      help='Number of samples to evaluate (None = all)')
    parser.add_argument('--log-interval', type=int, default=10,
                      help='Log TensorBoard images every N batches')
    parser.add_argument('--max-images', type=int, default=4,
                      help='Max images per visualization grid')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--strict-load', action='store_true',
                      help='Use strict mode when loading weights')

    return parser.parse_args()


def build_eval_model(args, device):
    """
    Build and load model for evaluation.

    For full-precision models: loads config from checkpoint.
    For quantized models: loads config from --config YAML, builds quantized model,
    then loads weights from checkpoint.

    Returns:
        (model, config) tuple
    """
    if args.quantized:
        # --- Quantized model ---
        if args.config is None:
            raise ValueError("--config is required when using --quantized")

        config = load_config(args.config)

        # Build model with quantization config
        model = get_model(config)

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            checkpoint = {'state_dict': state_dict}

        missing, unexpected = model.load_state_dict(state_dict, strict=args.strict_load)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        model = model.to(device)
        model.eval()

        # Set quantization mode to PTQ (inference)
        from snn.models.quant_utils import set_quant_mode, print_scale_summary
        set_quant_mode(model, 'ptq')
        print_scale_summary(model)

        quant_info = (f"W{config.get('weight_bit_width', 32)}"
                      f"A{config.get('act_bit_width', 32)}"
                      f"M{config.get('mem_bit_width', 32)}")
        print(f"Loaded quantized model ({quant_info}) from {args.checkpoint}")
        if 'epoch' in checkpoint:
            print(f"  Trained for {checkpoint['epoch']} epochs")
        if 'best_val_epe' in checkpoint:
            print(f"  Best validation EPE: {checkpoint['best_val_epe']:.4f}")

    else:
        # --- Full-precision model ---
        if args.config is not None:
            # Use external config but load weights from checkpoint
            config = load_config(args.config)
            model = get_model(config)

            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                checkpoint = {'state_dict': state_dict}

            model.load_state_dict(state_dict, strict=args.strict_load)
            model = model.to(device)
            model.eval()

            print(f"Loaded model from {args.checkpoint} (with external config)")
            if 'epoch' in checkpoint:
                print(f"  Trained for {checkpoint['epoch']} epochs")
            if 'best_val_epe' in checkpoint:
                print(f"  Best validation EPE: {checkpoint['best_val_epe']:.4f}")
        else:
            # Use config embedded in checkpoint
            model, config = build_model(None, device, train=False,
                                         checkpoint_path=args.checkpoint,
                                         strict=args.strict_load)

    model.disable_skip = True
    return model, config


def evaluate(args):
    """Main evaluation function"""

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Build model
    model, config = build_eval_model(args, device)

    # Build dataset
    data_root = args.data_root or config.get('val_data_root',
                    config.get('data_root', '../blink_sim/output/test_set'))
    dataset_config = config.copy()
    dataset_config['data_root'] = data_root
    dataset_config['max_train_samples'] = args.num_samples
    dataset_config['flip_left_to_right_prob'] = 0.0

    dataset = OpticalFlowDataset(config=dataset_config)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    print(f"Evaluating on {len(dataset)} samples from {data_root}")

    # Setup TensorBoard logger
    logger = Logger(log_dir=args.log_dir)
    logger.log_text('eval/checkpoint', args.checkpoint)
    logger.log_text('eval/data_root', data_root)
    logger.log_text('eval/quantized', str(args.quantized))
    if args.quantized:
        logger.log_text('eval/quant_config', args.config or 'embedded')

    vis_interval = args.log_interval
    max_vis_images = args.max_images
    num_bins = config.get('num_bins', 5)

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
            pred_flow = outputs.get('flow', outputs.get('pred_flow',
                list(outputs.values())[0])) if isinstance(outputs, dict) else outputs

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

            # ---- TensorBoard scalar per sample ----
            epe_val = metrics['epe']
            if isinstance(epe_val, torch.Tensor):
                epe_val = epe_val.detach().cpu().item()
            logger.log_scalar('eval/sample_epe', float(epe_val), idx)

            epe_mask_val = metrics['epe_mask']
            if isinstance(epe_mask_val, torch.Tensor):
                epe_mask_val = epe_mask_val.detach().cpu().item()
            logger.log_scalar('eval/sample_epe_masked', float(epe_mask_val), idx)

            # ---- TensorBoard visualizations ----
            # Use valid_mask (after low-activity masking) to match the trainer
            if idx % vis_interval == 0:
                bs = min(inputs.shape[0], max_vis_images)

                # Events (sum over polarities, keep time bins)
                event_sum = inputs[:bs].sum(dim=2, keepdim=True)
                event_vis = event_sum.repeat(1, 1, 3, 1, 1)
                grid = make_grid(event_vis.view(-1, 3, event_vis.shape[3], event_vis.shape[4]),
                                 nrow=num_bins, normalize=False, pad_value=1.0)
                logger.log_image('eval/events', grid, idx)

                # Valid mask (after low-activity masking, matching trainer)
                vm = valid_mask[:bs].repeat(1, 1, 3, 1, 1)
                grid = make_grid(vm.view(-1, 3, vm.shape[3], vm.shape[4]),
                                 nrow=num_bins, normalize=False, pad_value=1.0)
                logger.log_image('eval/valid_mask', grid, idx)

                max_flow = min(torch.norm(gt_flow, dim=1).max().item(), 1.0)

                # GT flow, predicted flow, and masked versions
                gt_vis, pred_vis = [], []
                gt_mask_vis, pred_mask_vis = [], []
                for i in range(bs):
                    gt_c = visualize_flow(gt_flow[i].cpu(), max_flow=max_flow)
                    gt_vis.append(torch.from_numpy(gt_c).permute(2, 0, 1).float() / 255.0)

                    pr_c = visualize_flow(pred_flow[i].cpu(), max_flow=max_flow)
                    pred_vis.append(torch.from_numpy(pr_c).permute(2, 0, 1).float() / 255.0)

                    gt_m = visualize_flow((gt_flow[i] * valid_mask[i]).cpu(), max_flow=max_flow)
                    gt_mask_vis.append(torch.from_numpy(gt_m).permute(2, 0, 1).float() / 255.0)

                    pr_m = visualize_flow((pred_flow[i] * valid_mask[i]).cpu(), max_flow=max_flow)
                    pred_mask_vis.append(torch.from_numpy(pr_m).permute(2, 0, 1).float() / 255.0)

                logger.log_image('eval/gt_flow',
                    make_grid(torch.stack(gt_vis), nrow=2, pad_value=1.0), idx)
                logger.log_image('eval/pred_flow',
                    make_grid(torch.stack(pred_vis), nrow=2, pad_value=1.0), idx)
                logger.log_image('eval/gt_flow_masked',
                    make_grid(torch.stack(gt_mask_vis), nrow=2, pad_value=1.0), idx)
                logger.log_image('eval/pred_flow_masked',
                    make_grid(torch.stack(pred_mask_vis), nrow=2, pad_value=1.0), idx)


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

    # Log summary scalars to TensorBoard
    logger.log_scalar('eval/avg_epe', avg_metrics['epe'], 0)
    logger.log_scalar('eval/avg_epe_masked', avg_metrics['epe_mask'], 0)
    logger.log_scalar('eval/avg_outliers', avg_metrics['outliers'], 0)
    logger.log_scalar('eval/avg_outliers_masked', avg_metrics['outliers_mask'], 0)
    logger.log_scalar('eval/avg_angular_error', avg_metrics['angular_error'], 0)
    logger.log_scalar('eval/avg_angular_error_masked', avg_metrics['angular_error_mask'], 0)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Checkpoint: {args.checkpoint}")
    if args.quantized:
        print(f"Quantized config: {args.config}")
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
        if args.quantized:
            f.write(f"Quantized config: {args.config}\n")
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
    print(f"TensorBoard logs saved to {args.log_dir}")
    print(f"  View with: tensorboard --logdir {args.log_dir}")



if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
