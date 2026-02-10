#!/usr/bin/env python3

"""Inspect a PyTorch checkpoint (.pth) used by this repo.

This prints a compact summary of:
- Top-level checkpoint keys and their types
- A best-effort detected state_dict (model weights) container
- Tensor entries: dtype, shape, number of elements
- Optional simple stats (min/max/mean/std) for tensors
- Membrane potential sizes (with --config option)

Examples:
  python tools/inspect_checkpoint.py checkpoints/full_p/best_model.pth
  python tools/inspect_checkpoint.py checkpoints/full_p/best_model.pth --stats
  python tools/inspect_checkpoint.py checkpoints/full_p/best_model.pth --grep "e1\\.conv"
  python tools/inspect_checkpoint.py checkpoints/full_p/best_model.pth --state-dict model_state_dict
  python tools/inspect_checkpoint.py checkpoints/full_p/best_model.pth --config snn/configs/event_snn_lite.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


DEFAULT_STATE_KEYS = ("model_state_dict", "state_dict", "model")


def _load_checkpoint(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def _is_tensor(obj: Any) -> bool:
    return isinstance(obj, torch.Tensor)


def _summarize_value(v: Any) -> str:
    if _is_tensor(v):
        t: torch.Tensor = v
        return f"Tensor(dtype={t.dtype}, shape={tuple(t.shape)}, numel={t.numel()})"
    if isinstance(v, (int, float, bool, str)):
        s = str(v)
        if len(s) > 80:
            s = s[:77] + "..."
        return f"{type(v).__name__}({s})"
    if isinstance(v, dict):
        return f"dict(len={len(v)})"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}(len={len(v)})"
    return type(v).__name__


def _choose_state_dict(ckpt: Any, forced_key: Optional[str]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if isinstance(ckpt, dict):
        if forced_key is not None:
            obj = ckpt.get(forced_key)
            if isinstance(obj, dict):
                return forced_key, obj
            return forced_key, None

        for k in DEFAULT_STATE_KEYS:
            obj = ckpt.get(k)
            if isinstance(obj, dict) and all(isinstance(kk, str) for kk in obj.keys()):
                return k, obj

        # Sometimes the checkpoint itself is a state_dict-like mapping.
        if all(isinstance(kk, str) for kk in ckpt.keys()):
            return "<root>", ckpt  # type: ignore[return-value]

    return None, None


@torch.no_grad()
def _tensor_stats(t: torch.Tensor, sample_max_elems: int = 2_000_000) -> str:
    # Avoid pulling giant tensors fully into reductions in case user runs on constrained machines.
    tt = t.detach().cpu()
    if tt.numel() == 0:
        return "min=nan max=nan mean=nan std=nan"

    if tt.numel() > sample_max_elems:
        flat = tt.flatten()
        step = max(1, flat.numel() // sample_max_elems)
        flat = flat[::step]
        tt = flat

    # Cast to float for stable stats
    tf = tt.float()
    if tt.numel() == 1:
        return f"val={tf.item():.6g}"
    else:
        return (
            f"min={tf.min().item():.6g} max={tf.max().item():.6g} "
            f"mean={tf.mean().item():.6g} std={tf.std(unbiased=False).item():.6g}"
        )


def _iter_items(d: Dict[str, Any], key_re: Optional[re.Pattern[str]]) -> Iterable[Tuple[str, Any]]:
    for k in sorted(d.keys()):
        if key_re is not None and key_re.search(k) is None:
            continue
        yield k, d[k]


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint (.pth)")
    ap.add_argument("checkpoint", type=str, help="Path to .pth file")
    ap.add_argument(
        "--list-top",
        action="store_true",
        help="List top-level keys/types (default: enabled)",
    )
    ap.add_argument(
        "--no-list-top",
        action="store_true",
        help="Disable listing top-level keys",
    )
    ap.add_argument(
        "--state-dict",
        type=str,
        default=None,
        help="Force which top-level key contains the state_dict (e.g. model_state_dict)",
    )
    ap.add_argument(
        "--grep",
        type=str,
        default=None,
        help="Regex to filter state_dict keys",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Limit number of printed state_dict entries",
    )
    ap.add_argument(
        "--stats",
        default=True,
        action="store_true",
        help="Print simple tensor stats (min/max/mean/std)",
    )
    ap.add_argument(
        "--show-non-tensors",
        default=True,
        action="store_true",
        help="Also show non-tensor entries inside the chosen dict",
    )
    ap.add_argument(
        "--config",
        type=str,
        default="../snn/configs/event_snn_lite.yaml",
        help="Path to model config YAML file (required for membrane size calculation)",
    )
    ap.add_argument(
        "--input-size",
        type=str,
        default="320,320",
        help="Input image size as 'H,W' for membrane calculation (default: 320,320)",
    )

    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = _load_checkpoint(ckpt_path)

    print(f"[ckpt] path: {ckpt_path}")
    print(f"[ckpt] type: {type(ckpt).__name__}")

    list_top = True
    if args.no_list_top:
        list_top = False
    if args.list_top:
        list_top = True

    if list_top and isinstance(ckpt, dict):
        print("\n[top-level keys]")
        for k in sorted(ckpt.keys()):
            v = ckpt[k]
            print(f"- {k}: {_summarize_value(v)}")

    state_key, sd = _choose_state_dict(ckpt, args.state_dict)
    if sd is None:
        print("\n[state_dict]")
        if args.state_dict is not None:
            print(f"- Could not find dict at key '{args.state_dict}'")
        else:
            print(f"- Could not auto-detect state_dict. Tried: {', '.join(DEFAULT_STATE_KEYS)}")
        return

    key_re = re.compile(args.grep) if args.grep else None

    num_tensors = 0
    num_other = 0
    for _, v in sd.items():
        if _is_tensor(v):
            num_tensors += 1
        else:
            num_other += 1

    print("\n[state_dict]")
    print(f"- selected: {state_key}")
    print(f"- entries: {len(sd)} (tensors={num_tensors}, other={num_other})")

    print("\n[state_dict entries]")
    shown = 0
    sum = 0
    for k, v in _iter_items(sd, key_re):
        if _is_tensor(v):
            t: torch.Tensor = v
            base = f"- {k}: dtype={t.dtype} shape={tuple(t.shape)} numel={t.numel()}"
            if args.stats:
                base += " " + _tensor_stats(t)
            print(base)
            shown += 1
            sum += t.numel()
        else:
            if args.show_non_tensors:
                print(f"- {k}: {_summarize_value(v)}")
                shown += 1

        if shown >= args.limit:
            remaining = max(0, len(sd) - shown)
            if remaining:
                print(f"... (stopped at --limit={args.limit}, remaining keys not shown)")
            break
    print(f"\nTotal parameters in shown entries: {sum}")
    
    # Calculate membrane potential sizes if config is provided
    if args.config is not None:
        _calculate_membrane_sizes(ckpt, args.config, args.input_size)


def _calculate_membrane_sizes(ckpt: Any, config_path: str, input_size_str: str) -> None:
    """Calculate total size of membrane potentials by running a dummy forward pass."""
    print("\n[membrane potential sizes]")
    
    try:
        # Add parent directory to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils import load_config, build_model
        
        # Load config
        config = load_config(config_path)
        
        # Parse input size
        try:
            h, w = map(int, input_size_str.split(','))
        except ValueError:
            print(f"- Error: Invalid input size format '{input_size_str}'. Expected 'H,W'.")
            return
        
        # Build model
        try:
            model = build_model(config)
            
            # Load checkpoint weights
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                print("- Warning: Could not find model_state_dict or state_dict in checkpoint")
            
            model.eval()
        except Exception as e:
            print(f"- Error building model: {e}")
            return
        
        # Create dummy input
        num_bins = config.get('num_bins', 5)
        use_polarity = config.get('use_polarity', False)
        c = 2 if use_polarity else 1
        
        dummy_input = torch.zeros(1, num_bins, c, h, w, device='cuda')
        print(f"- input shape: [1, {num_bins}, {c}, {h}, {w}]")
        
        # Hook to capture membrane states
        membrane_states = {}
        
        def capture_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    # Output is (spikes, membrane)
                    mem = output[1]
                    if mem is not None and isinstance(mem, torch.Tensor):
                        # Store shape without batch dimension
                        membrane_states[name] = tuple(mem.shape[1:])
            return hook
        
        # Register hooks on encoder/decoder layers
        hooks = []
        for name, module in model.named_modules():
            if any(x in name for x in ['e1', 'e2', 'e3', 'e4', 'd1', 'd2', 'd3', 'd4']):
                if 'conv' in name or name in ['e1', 'e2', 'e3', 'e4', 'd1', 'd2', 'd3', 'd4']:
                    hook = module.register_forward_hook(capture_hook(name))
                    hooks.append(hook)
        
        # Run forward pass
        try:
            with torch.no_grad():
                _ = model(dummy_input)
        except Exception as e:
            print(f"- Error during forward pass: {e}")
            for hook in hooks:
                hook.remove()
            return
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate and display sizes
        if membrane_states:
            print("\n- membrane states:")
            total_elements = 0
            total_bytes = 0
            
            # Sort by layer name for consistent output
            for name in sorted(membrane_states.keys()):
                shape = membrane_states[name]
                elements = 1
                for dim in shape:
                    elements *= dim
                
                # Assume float32 (4 bytes per element)
                bytes_size = elements * 1
                total_elements += elements
                total_bytes += bytes_size
                
                # Format size nicely
                if bytes_size < 1024:
                    size_str = f"{bytes_size} B"
                elif bytes_size < 1024**2:
                    size_str = f"{bytes_size/1024:.2f} KB"
                elif bytes_size < 1024**3:
                    size_str = f"{bytes_size/(1024**2):.2f} MB"
                else:
                    size_str = f"{bytes_size/(1024**3):.2f} GB"
                
                print(f"  {name}: {shape} -> {elements:,} elements ({size_str})")
            
            # Total summary
            print(f"\n- total membrane elements: {total_elements:,}")
            if total_bytes < 1024:
                total_str = f"{total_bytes} B"
            elif total_bytes < 1024**2:
                total_str = f"{total_bytes/1024:.2f} KB"
            elif total_bytes < 1024**3:
                total_str = f"{total_bytes/(1024**2):.2f} MB"
            else:
                total_str = f"{total_bytes/(1024**3):.2f} GB"
            print(f"- total membrane memory: {total_str}")
        else:
            print("- No membrane states captured (model may not have SNN layers)")
            
    except ImportError as e:
        print(f"- Error importing required modules: {e}")
        print("- Make sure you run this from the project root directory")
    except Exception as e:
        print(f"- Error: {e}")


if __name__ == "__main__":
    main()
