#!/usr/bin/env python3
"""
Examine a trained model — works with both full-precision and quantized checkpoints.

Provides a unified view of:
  - Model architecture summary (layers, parameters, sizes)
  - Quantization scale summary (if quantized)
  - Exported integer weights & per-layer bit widths (if quantized)
  - Weight statistics (min/max/mean/std per layer)
  - Activation range estimates from calibration (if quantized)

Does NOT require a GPU — everything runs on CPU.

Examples:
    # Full-precision model (config embedded in checkpoint)
    python tools/examine_model.py checkpoints/teacher_10000u/best_model.pth

    # Quantized model (needs config for quant settings)
    python tools/examine_model.py checkpoints/ptq_8bit/ptq_model.pth \\
        --config snn/configs/event_snn_lite_8bit.yaml --quantized

    # Export integer weights to a .pt file for FPGA tooling
    python tools/examine_model.py checkpoints/ptq_8bit/ptq_model.pth \\
        --config snn/configs/event_snn_lite_8bit.yaml --quantized \\
        --export exported_weights.pt

    # Filter layers by regex
    python tools/examine_model.py checkpoints/ptq_8bit/ptq_model.pth \\
        --config snn/configs/event_snn_lite_8bit.yaml --quantized \\
        --grep "e1\\|d1"

    # Calibrate a full-precision model with PTQ and inspect scales
    python tools/examine_model.py checkpoints/teacher_10000u/best_model.pth \\
        --config snn/configs/event_snn_lite_8bit.yaml --quantized \\
        --calibrate --data-root ../blink_sim/output/train_set
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import load_config, get_model, build_model
from snn.models.quant_utils import (
    QuantWeight, QuantAct,
    print_scale_summary,
    export_quantized_params,
    set_quant_mode,
    calibrate_model,
)


# ============================================================================
# Model loading
# ============================================================================

def load_model(checkpoint_path: str, config_path: Optional[str], quantized: bool,
               device: str = 'cpu') -> tuple:
    """Load model from checkpoint, returning (model, config, checkpoint_meta)."""

    if quantized:
        if config_path is None:
            raise ValueError("--config is required when using --quantized")

        config = load_config(config_path)
        model = get_model(config)

        ckpt = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
            ckpt = {'state_dict': sd}

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        model.eval()
        set_quant_mode(model, 'ptq')
        return model, config, ckpt

    else:
        if config_path is not None:
            config = load_config(config_path)
            model = get_model(config)
            ckpt = torch.load(checkpoint_path, map_location=device)
            sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            model.load_state_dict(sd, strict=False)
            model.eval()
            return model, config, ckpt
        else:
            model, config = build_model(None, device, train=False,
                                         checkpoint_path=checkpoint_path, strict=False)
            ckpt = torch.load(checkpoint_path, map_location=device)
            return model, config, ckpt


# ============================================================================
# Inspection helpers
# ============================================================================

def print_checkpoint_meta(ckpt: dict) -> None:
    """Print training metadata from checkpoint."""
    print("\n" + "=" * 70)
    print("Checkpoint Metadata")
    print("=" * 70)
    for key in ('epoch', 'best_val_epe', 'global_step', 'mode', 'pretrained_from'):
        if key in ckpt:
            print(f"  {key:20s}: {ckpt[key]}")
    if 'config' in ckpt:
        cfg = ckpt['config']
        print(f"  {'model_type':20s}: {cfg.get('model_type', '?')}")
        print(f"  {'base_ch':20s}: {cfg.get('base_ch', '?')}")
        if cfg.get('quantize_weights') or cfg.get('quantize_activations'):
            print(f"  {'quant':20s}: W{cfg.get('weight_bit_width', '?')}"
                  f"A{cfg.get('act_bit_width', '?')}"
                  f"M{cfg.get('mem_bit_width', '?')}")


def print_architecture_summary(model: nn.Module, grep: Optional[str] = None) -> None:
    """Print a layer-by-layer summary of the model architecture."""
    print("\n" + "=" * 70)
    print("Architecture Summary")
    print("=" * 70)

    pattern = re.compile(grep, re.IGNORECASE) if grep else None

    total_params = 0
    trainable_params = 0

    header = f"  {'Layer':<45s} {'Type':<25s} {'Params':>10s} {'Shape'}"
    print(header)
    print("  " + "-" * len(header))

    for name, param in model.named_parameters():
        if pattern and not pattern.search(name):
            continue
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
        print(f"  {name:<45s} {str(tuple(param.shape)):<25s} {numel:>10,d}")

    print("  " + "-" * len(header))
    print(f"  {'Total parameters':45s} {'':25s} {total_params:>10,d}")
    print(f"  {'Trainable':45s} {'':25s} {trainable_params:>10,d}")
    print(f"  {'Frozen':45s} {'':25s} {total_params - trainable_params:>10,d}")


def print_weight_statistics(model: nn.Module, grep: Optional[str] = None) -> None:
    """Print min/max/mean/std of weight tensors."""
    print("\n" + "=" * 70)
    print("Weight Statistics")
    print("=" * 70)

    pattern = re.compile(grep, re.IGNORECASE) if grep else None

    header = f"  {'Name':<45s} {'Shape':<18s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'Std':>10s}"
    print(header)
    print("  " + "-" * len(header))

    with torch.no_grad():
        for name, param in model.named_parameters():
            if pattern and not pattern.search(name):
                continue
            if 'weight' not in name and 'bias' not in name:
                continue
            t = param.float()
            print(f"  {name:<45s} {str(tuple(t.shape)):<18s} "
                  f"{t.min().item():>10.5f} {t.max().item():>10.5f} "
                  f"{t.mean().item():>10.5f} {t.std().item():>10.5f}")


def print_quant_module_details(model: nn.Module, grep: Optional[str] = None) -> None:
    """Print detailed info about every QuantWeight and QuantAct module."""
    print("\n" + "=" * 70)
    print("Quantizer Module Details")
    print("=" * 70)

    pattern = re.compile(grep, re.IGNORECASE) if grep else None

    has_any = False
    for name, module in model.named_modules():
        if pattern and not pattern.search(name):
            continue

        if isinstance(module, QuantWeight):
            has_any = True
            s = module.scale
            cal = "yes" if module._calibrated else "no"
            mode = "ptq" if module._ptq_mode else "qat"
            print(f"\n  [QuantWeight] {name}")
            print(f"    bit_width:  {module.bit_width}")
            print(f"    scale_type: {module.scale_type}")
            print(f"    mode:       {mode}")
            print(f"    calibrated: {cal}")
            if s.numel() == 1:
                print(f"    scale:      {s.item():.8f}")
            else:
                print(f"    scale:      [{s.min().item():.8f}, {s.max().item():.8f}]  "
                      f"(shape={tuple(s.shape)}, mean={s.mean().item():.8f})")

        elif isinstance(module, QuantAct):
            has_any = True
            s = module.scale
            zp = module.zero_point
            cal = "yes" if module._calibrated else "no"
            mode = "ptq" if module._ptq_mode else "qat"
            print(f"\n  [QuantAct] {name}")
            print(f"    bit_width:   {module.bit_width}")
            print(f"    scale_type:  {module.scale_type}")
            print(f"    symmetric:   {module.symmetric}")
            print(f"    mode:        {mode}")
            print(f"    calibrated:  {cal}")
            print(f"    batches_seen:{module._num_batches_seen}")
            if s.numel() == 1:
                print(f"    scale:       {s.item():.8f}")
                print(f"    zero_point:  {zp.item():.1f}")
            else:
                print(f"    scale:       [{s.min().item():.8f}, {s.max().item():.8f}]  "
                      f"(shape={tuple(s.shape)})")
                print(f"    zero_point:  [{zp.min().item():.1f}, {zp.max().item():.1f}]")
            print(f"    running_min: {module.running_min.min().item():.6f}")
            print(f"    running_max: {module.running_max.max().item():.6f}")

    if not has_any:
        print("  No QuantWeight/QuantAct modules found — model is full-precision.")


def print_export_summary(exported: Dict[str, Dict], grep: Optional[str] = None) -> None:
    """Print a summary of exported integer parameters."""
    print("\n" + "=" * 70)
    print("Exported Quantized Parameters")
    print("=" * 70)

    pattern = re.compile(grep, re.IGNORECASE) if grep else None

    for layer_name, info in exported.items():
        if pattern and not pattern.search(layer_name):
            continue
        parts = [f"  {layer_name}"]

        if 'int_weight' in info:
            iw = info['int_weight']
            wb = info.get('weight_bit_width', '?')
            ws = info['weight_scale']
            parts.append(f"    Weight: int{wb} shape={tuple(iw.shape)} "
                         f"range=[{iw.min().item()}, {iw.max().item()}]")
            if ws.numel() == 1:
                parts.append(f"    Weight scale (S_w): {ws.item():.8f}")
            else:
                parts.append(f"    Weight scale (S_w): [{ws.min().item():.8f}, {ws.max().item():.8f}]")

        if 'act_scale' in info:
            ab = info.get('act_bit_width', '?')
            a_s = info['act_scale']
            a_zp = info.get('act_zero_point', torch.tensor(0))
            if a_s.numel() == 1:
                parts.append(f"    Act (S_out): {ab}-bit  scale={a_s.item():.8f}  zp={a_zp.item():.1f}")
            else:
                parts.append(f"    Act (S_out): {ab}-bit  scale=[{a_s.min().item():.8f}, "
                             f"{a_s.max().item():.8f}]")

        if 'input_scale' in info:
            s_in = info['input_scale']
            if s_in.numel() == 1:
                parts.append(f"    Input scale (S_in): {s_in.item():.8f}")
            else:
                parts.append(f"    Input scale (S_in): [{s_in.min().item():.8f}, {s_in.max().item():.8f}]")

        if 'M_real' in info:
            m_r = info['M_real']
            m0 = info['M_0']
            sh = info['shift']
            mb = info.get('multiplier_bits', '?')
            if m_r.numel() == 1:
                parts.append(f"    M_real = S_w*S_in/S_out = {m_r.item():.8f}")
                parts.append(f"    M_0 = {m0.item()}  shift = {sh.item()}  "
                             f"({mb}-bit multiplier)")
                # Verify
                approx = m0.item() * (2.0 ** (-sh.item()))
                parts.append(f"    Verify: M_0 * 2^(-shift) = {approx:.8f}  "
                             f"(error = {abs(approx - m_r.item()):.2e})")
            else:
                parts.append(f"    M_real: [{m_r.min().item():.8f}, {m_r.max().item():.8f}]  "
                             f"(shape={tuple(m_r.shape)})")
                parts.append(f"    M_0:    [{m0.min().item()}, {m0.max().item()}]  "
                             f"shift: [{sh.min().item()}, {sh.max().item()}]  "
                             f"({mb}-bit multiplier)")

        for p in parts:
            print(p)
        print()

    print(f"  Total exported layers: {len(exported)}")
    print(f"\n  Note: S_in = 1.0 for SNN layers where the input is binary spikes.")
    print(f"  FPGA integer inference: out_int = (M_0 * acc_int) >> shift")


# ============================================================================
# Calibration helper
# ============================================================================

def run_calibration(model: nn.Module, config: dict, data_root: str,
                    num_batches: int = 50) -> None:
    """Run PTQ calibration on a few batches of training data."""
    from torch.utils.data import DataLoader
    from snn.dataset import OpticalFlowDataset

    cal_config = config.copy()
    cal_config['data_root'] = data_root

    dataset = OpticalFlowDataset(config=cal_config)
    loader = DataLoader(dataset, batch_size=config.get('batch_size', 4),
                        shuffle=False, num_workers=2)

    print(f"\n[Calibration] Running {num_batches} batches from {data_root} ...")
    calibrate_model(model, loader, device='cpu', num_batches=num_batches,
                    config=config)
    print("[Calibration] Done.")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Examine a trained SNN model (full-precision or quantized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('checkpoint', type=str,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML (required for --quantized)')
    parser.add_argument('--quantized', action='store_true',
                        help='Treat as a quantized model')

    # What to show
    parser.add_argument('--no-arch', action='store_true',
                        help='Skip architecture summary')
    parser.add_argument('--no-stats', action='store_true',
                        help='Skip weight statistics')
    parser.add_argument('--grep', type=str, default=None,
                        help='Regex filter for layer names')

    # Quantization-specific
    parser.add_argument('--export', type=str, default=None,
                        help='Export integer weights to this .pt file')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run PTQ calibration before inspection')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Data root for calibration (--calibrate)')
    parser.add_argument('--num-calib-batches', type=int, default=50,
                        help='Number of calibration batches (default 50)')

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, config, ckpt = load_model(args.checkpoint, args.config, args.quantized)

    # ---- Checkpoint metadata ----
    print_checkpoint_meta(ckpt)

    # ---- Optional calibration ----
    if args.calibrate:
        data_root = args.data_root or config.get('data_root',
                        '../blink_sim/output/train_set')
        num_batches = args.num_calib_batches or config.get('num_calib_batches', 50)
        run_calibration(model, config, data_root, num_batches)

    # ---- Architecture ----
    if not args.no_arch:
        print_architecture_summary(model, grep=args.grep)

    # ---- Weight statistics ----
    if not args.no_stats:
        print_weight_statistics(model, grep=args.grep)

    # ---- Quantization info ----
    if args.quantized:
        print_scale_summary(model)
        print_quant_module_details(model, grep=args.grep)

        # Export
        exported = export_quantized_params(model)
        if exported:
            print_export_summary(exported, grep=args.grep)

            if args.export:
                torch.save(exported, args.export)
                print(f"\nExported integer weights saved to: {args.export}")
        else:
            print("\n  No exportable quantized layers found.")

    print()


if __name__ == '__main__':
    main()
