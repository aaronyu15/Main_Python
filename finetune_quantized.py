"""
PTQ Calibration Script

Post-Training Quantization:
  - Load pretrained model, run calibration pass, export quantized model
  - No training required — just calibration over a few batches of data

Usage:
    # Basic PTQ calibration:
    python finetune_quantized.py \
        --config snn/configs/event_snn_lite_8bit.yaml \
        --pretrained checkpoints/02_no_d1/best_model.pth \
        --name 02_no_d1

    # With integer inference export (saves integer params + model report):
    python finetune_quantized.py \
        --config snn/configs/event_snn_lite_8bit.yaml \
        --pretrained checkpoints/02_no_d1/best_model.pth \
        --name 02_no_d1 \
        --export-integer
"""
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from snn.models import EventSNNFlowNetLite
from snn.dataset import OpticalFlowDataset
from snn.training import SNNTrainer
from snn.utils.logger import Logger
from snn.models.quant_utils import (
    enable_quantization_warnings, reset_quantization_warning_counts,
    calibrate_model, print_scale_summary, export_quantized_params,
    reset_all_overflow_trackers, log_all_overflow_stats, print_overflow_summary,
)
from snn.utils.visualization import visualize_flow
from torchvision.utils import make_grid
import random
import warnings

from utils import load_config, get_model


def parse_args():
    parser = argparse.ArgumentParser(description='PTQ Calibration for SNN')
    
    # Config
    parser.add_argument('--config', type=str, required=True,
                      help='Path to quantized model configuration file')
    
    # Pretrained model
    parser.add_argument('--pretrained', type=str, required=True,
                      help='Path to pre-trained model checkpoint to calibrate')
    parser.add_argument('--name', type=str, required=True,
                      help='Path to pre-trained model checkpoint to calibrate')
    
    # Training settings
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume fine-tuning from')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                      help='Directory for logs')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Data settings
    parser.add_argument('--train-data-root', type=str, default=None,
                      help='Override training dataset root')
    parser.add_argument('--val-data-root', type=str, default=None,
                      help='Validation dataset root')
    
    # Quantization warnings
    parser.add_argument('--disable-quant-warnings', action='store_true',
                      help='Disable quantization range warnings')
    parser.add_argument('--strict-load', action='store_true',
                      help='Use strict mode when loading pretrained weights')
    parser.add_argument('--export-integer', action='store_true',
                      help='Export integer inference parameters into checkpoint and generate model report')
    
    return parser.parse_args()


def build_dataloaders(config: dict, train_root: str = None, val_root: str = None):
    """Build training and validation data loaders."""
    train_root = train_root or config.get('data_root', '../blink_sim/output/train_set')
    val_root = val_root or config.get('val_data_root', '../blink_sim/output/valid_set')

    # Train dataset
    train_config = config.copy()
    train_config['data_root'] = train_root
    train_dataset = OpticalFlowDataset(config=train_config)

    # Val dataset
    val_config = config.copy()
    val_config['data_root'] = val_root
    val_dataset = OpticalFlowDataset(config=val_config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader


def load_pretrained_weights(model: nn.Module, checkpoint_path: str, strict: bool = False) -> dict:
    """
    Load pretrained weights into the model.
    
    Handles weight loading from full-precision or different bit-width models.
    Uses non-strict loading by default to handle quantization-related parameter differences.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to pretrained checkpoint
        strict: Whether to require exact match of state dict keys
        
    Returns:
        Checkpoint dict containing metadata
    """
    print(f"Loading pretrained weights from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume checkpoint is the state dict itself
        state_dict = checkpoint
        checkpoint = {'state_dict': state_dict}
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"  Pretrained model trained for {checkpoint['epoch']} epochs")
    if 'best_val_epe' in checkpoint:
        print(f"  Pretrained best validation EPE: {checkpoint['best_val_epe']:.4f}")
    if 'config' in checkpoint:
        pretrained_config = checkpoint['config']
        print(f"  Pretrained quantization: W{pretrained_config.get('weight_bit_width', 32)}A{pretrained_config.get('act_bit_width', 32)}M{pretrained_config.get('mem_bit_width', 32)}")
    
    return checkpoint


def print_quantization_info(config: dict):
    """Print quantization settings."""
    print("\n" + "="*60)
    print("Quantization-Aware Fine-tuning Configuration")
    print("="*60)
    
    quant_weights = config.get('quantize_weights', False)
    quant_act = config.get('quantize_activations', False)
    quant_mem = config.get('quantize_mem', False)
    
    weight_bits = config.get('weight_bit_width', 8)
    act_bits = config.get('act_bit_width', 8)
    mem_bits = config.get('mem_bit_width', 16)
    
    print(f"Weight quantization:     {'Enabled' if quant_weights else 'Disabled'} ({weight_bits}-bit)")
    print(f"Activation quantization: {'Enabled' if quant_act else 'Disabled'} ({act_bits}-bit)")
    print(f"Membrane quantization:   {'Enabled' if quant_mem else 'Disabled'} ({mem_bits}-bit)")
    
    if quant_weights or quant_act or quant_mem:
        # Calculate theoretical compression
        if quant_weights:
            compression = 32.0 / weight_bits
            print(f"\nTheoretical weight compression: {compression:.1f}x")
        
        # Print quantization ranges
        print(f"\nQuantization ranges:")
        if quant_weights:
            qmax_w = 2 ** (weight_bits - 1) - 1
            print(f"  Weights ({weight_bits}-bit symmetric): [{-qmax_w-1}, {qmax_w}]")
        if quant_act:
            qmax_a = 2 ** act_bits - 1
            print(f"  Activations ({act_bits}-bit asymmetric): [0, {qmax_a}]")
        if quant_mem:
            qmax_m = 2 ** (mem_bits - 1) - 1
            print(f"  Membrane ({mem_bits}-bit symmetric): [{-qmax_m-1}, {qmax_m}]")
    
    print("="*60 + "\n")


def _fmt_tensor_1d(t, max_per_line=16):
    """Format a 1D tensor as a compact string."""
    vals = t.view(-1).tolist()
    if len(vals) <= max_per_line:
        return '[' + ', '.join(f'{v:g}' for v in vals) + ']'
    lines = []
    for i in range(0, len(vals), max_per_line):
        chunk = vals[i:i+max_per_line]
        lines.append('  ' + ', '.join(f'{v:g}' for v in chunk))
    return '[\n' + ',\n'.join(lines) + '\n]'


def _fmt_kernel(t):
    """Format a 2D kernel (e.g. 3×3) as a readable grid."""
    rows = []
    for r in range(t.shape[0]):
        row_vals = ', '.join(f'{v:8.4f}' for v in t[r].tolist())
        rows.append(f'  [{row_vals}]')
    return '[\n' + '\n'.join(rows) + '\n]'


def _fmt_kernel_int(t):
    """Format a 2D integer kernel as a readable grid."""
    rows = []
    for r in range(t.shape[0]):
        row_vals = ', '.join(f'{v:5d}' for v in t[r].tolist())
        rows.append(f'  [{row_vals}]')
    return '[\n' + '\n'.join(rows) + '\n]'


def generate_model_report(exported, config, output_path, pretrained_from=None):
    """Generate a comprehensive text file detailing the model structure and parameters."""
    import datetime
    lines = []
    w = lines.append

    w('=' * 72)
    w('SNN Optical Flow Model — Comprehensive Parameter Report')
    w('=' * 72)
    w(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    if pretrained_from:
        w(f'Pretrained from: {pretrained_from}')
    w('')

    # --- Configuration ---
    w('=' * 72)
    w('MODEL CONFIGURATION')
    w('=' * 72)
    for key in ['model_type', 'base_ch', 'use_polarity', 'num_bins',
                'weight_bit_width', 'act_bit_width', 'mem_bit_width',
                'accum_bit_width', 'multiplier_bits',
                'weight_scale_type', 'act_scale_type',
                'quantize_weights', 'quantize_activations', 'quantize_mem']:
        val = config.get(key, '—')
        w(f'  {key}: {val}')
    w('')

    # --- Network Structure Overview ---
    w('=' * 72)
    w('NETWORK STRUCTURE OVERVIEW')
    w('=' * 72)
    header = f'  {"Layer":<15s} {"Weight Shape":<18s} {"Stride":>6s} {"Pad":>4s} {"Groups":>6s} {"LIF Type"}'
    w(header)
    w('  ' + '-' * 70)

    for conv_name, params in exported.items():
        name = conv_name.rsplit('.', 1)[0] if '.' in conv_name else conv_name
        if 'int_weight' not in params:
            continue
        shape_str = 'x'.join(str(d) for d in params['int_weight'].shape)
        stride = params.get('stride', '?')
        padding = params.get('padding', '?')
        groups = params.get('groups', '?')
        if 'lif_type' in params:
            lif_str = params['lif_type']
            opt = params.get('lif_option')
            if opt:
                lif_str += f' ({opt})'
        else:
            lif_str = '— (no LIF)'
        w(f'  {name:<15s} {shape_str:<18s} {stride:>6} {padding:>4} {groups:>6} {lif_str}')
    w('')

    # --- Detailed Layer Parameters ---
    w('=' * 72)
    w('DETAILED LAYER PARAMETERS')
    w('=' * 72)

    for conv_name, params in exported.items():
        name = conv_name.rsplit('.', 1)[0] if '.' in conv_name else conv_name
        if 'int_weight' not in params:
            continue

        w('')
        w('─' * 72)
        w(f'Layer: {conv_name}')
        if 'lif_type' in params:
            w(f'Block: SpikingConvBlock (QuantizedConv2d + {params["lif_type"]})')
        else:
            w(f'Block: ConvBlock (QuantizedConv2d, no LIF)')
        w('─' * 72)

        wt = params['int_weight']
        w(f'  Weight shape: {list(wt.shape)} ({wt.numel()} parameters)')
        w(f'  Stride: {params.get("stride", "?")}, '
          f'Padding: {params.get("padding", "?")}, '
          f'Groups: {params.get("groups", "?")}')
        w(f'  Weight bit width: {params.get("weight_bit_width", "?")}')
        w(f'  Activation bit width: {params.get("act_bit_width", "?")}')

        # Scales
        w('')
        w('  --- Quantization Scales ---')
        w(f'  Weight scale (S_w): {_fmt_tensor_1d(params["weight_scale"])}')
        w(f'  Input scale (S_in): {_fmt_tensor_1d(params.get("input_scale", torch.tensor(1.0)))}')
        if 'act_scale' in params:
            w(f'  Output activation scale (S_out): {_fmt_tensor_1d(params["act_scale"])}')

        # Requantization
        w('')
        w('  --- Requantization: out = (M_0 * acc) >> shift ---')
        w(f'  M_real:  {_fmt_tensor_1d(params["M_real"])}')
        w(f'  M_0:     {_fmt_tensor_1d(params["M_0"])}')
        w(f'  Shift:   {_fmt_tensor_1d(params["shift"])}')
        w(f'  Multiplier bits: {params.get("multiplier_bits", "?")}')

        # LIF
        if 'lif_type' in params:
            w('')
            w('  --- LIF Neuron ---')
            w(f'  Type: {params["lif_type"]}')
            w(f'  Decay: {params.get("decay", "None")}')
            w(f'  Option: {params.get("lif_option", "None")}')
            w(f'  Threshold (float): {params["threshold_float"]}')
            w(f'  Threshold (int):   {_fmt_tensor_1d(params["threshold_int"])}')

        # Float weights
        w('')
        w('  --- Float Weights ---')
        float_w = params.get('float_weight', None)
        if float_w is not None:
            Cout, Cin_g, kH, kW = float_w.shape
            for ic in range(Cin_g):
                for oc in range(Cout):
                    w(f'  Input ch {ic}, Filter {oc}:')
                    w(f'  {_fmt_kernel(float_w[oc, ic])}')
        else:
            w('  (not available)')

        # Integer weights
        w('')
        w('  --- Integer Weights ---')
        Cout, Cin_g, kH, kW = wt.shape
        for ic in range(Cin_g):
            for oc in range(Cout):
                w(f'  Input ch {ic}, Filter {oc}:')
                w(f'  {_fmt_kernel_int(wt[oc, ic])}')

    w('')
    w('=' * 72)
    w('END OF REPORT')
    w('=' * 72)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Setup quantization warnings
    if args.disable_quant_warnings:
        enable_quantization_warnings(False)
        print("Quantization range warnings disabled")
    else:
        enable_quantization_warnings(True)
        reset_quantization_warning_counts()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed: {args.seed}")
    
    # Print quantization info
    print_quantization_info(config)
    
    # Build model with quantization config
    model = get_model(config)
    print(f"Built model: {config.get('model_type', 'EventSNNFlowNetLite')}")
    
    # Load pretrained weights
    pretrained_checkpoint = load_pretrained_weights(
        model, 
        args.pretrained, 
        strict=args.strict_load
    )
    
    # Move to device
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    checkpoint_dir = Path(args.checkpoint_dir) / args.name
    log_dir = Path(args.log_dir) / args.name
    # Logger
    logger = Logger(log_dir=log_dir)
    print(log_dir)
    model.set_logger(logger)
    
    # Build dataloaders
    train_root = args.train_data_root or config.get('data_root', None)
    val_root = args.val_data_root or config.get('val_data_root', None)
    train_loader, val_loader = build_dataloaders(config, train_root, val_root)

    print(f"\nData:")
    print(f"  Train data root: {train_root}")
    print(f"  Val data root: {val_root}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Log configuration
    logger.log_config(config, model=model)
    
    # Add quantization info to logged config
    logger.log_text('quantization/pretrained_from', args.pretrained)
    logger.log_text('quantization/weight_bits', str(config.get('weight_bit_width', 8)))
    logger.log_text('quantization/act_bits', str(config.get('act_bit_width', 8)))
    logger.log_text('quantization/mem_bits', str(config.get('mem_bit_width', 16)))
    
    # ---- PTQ: calibrate and save ----
    print(f"\n[PTQ] Running post-training quantization...")
    
    calibrate_model(
        model, train_loader,
        device=device,
        num_batches=config.get('num_calib_batches', 100),
        config=config,
    )
    
    # Save calibrated model
    save_path = checkpoint_dir
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'mode': 'ptq',
        'pretrained_from': args.pretrained,
    }, save_path / 'ptq_model.pth')
    
    print(f"\n[PTQ] Calibrated model saved to: {save_path / 'ptq_model.pth'}")

    # Optionally export integer inference parameters
    if args.export_integer:
        multiplier_bits = config.get('multiplier_bits', 16)
        print(f"\n[Export] Exporting integer inference parameters (multiplier_bits={multiplier_bits})...")
        exported = export_quantized_params(model, multiplier_bits=multiplier_bits)

        # Re-save checkpoint with integer params included
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'mode': 'ptq',
            'pretrained_from': args.pretrained,
            'integer_params': exported,
        }, save_path / 'ptq_model.pth')
        print(f"[Export] Saved checkpoint with integer params: {save_path / 'ptq_model.pth'}")

        # Generate comprehensive text report
        report_path = save_path / 'model_report.txt'
        generate_model_report(exported, config, report_path, args.pretrained)
        print(f"[Export] Model report saved to: {report_path}")


if __name__ == '__main__':
    main()
