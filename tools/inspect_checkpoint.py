#!/usr/bin/env python3

"""Inspect a PyTorch checkpoint (.pth) used by this repo.

This prints a compact summary of:
- Top-level checkpoint keys and their types
- A best-effort detected state_dict (model weights) container
- Tensor entries: dtype, shape, number of elements
- Optional simple stats (min/max/mean/std) for tensors

Examples:
  python tools/inspect_checkpoint.py checkpoints/full_p/best_model.pth
  python tools/inspect_checkpoint.py checkpoints/full_p/best_model.pth --stats
  python tools/inspect_checkpoint.py checkpoints/full_p/best_model.pth --grep "e1\\.conv"
  python tools/inspect_checkpoint.py checkpoints/full_p/best_model.pth --state-dict model_state_dict
"""

from __future__ import annotations

import argparse
import re
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
        action="store_true",
        help="Print simple tensor stats (min/max/mean/std)",
    )
    ap.add_argument(
        "--show-non-tensors",
        action="store_true",
        help="Also show non-tensor entries inside the chosen dict",
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
    for k, v in _iter_items(sd, key_re):
        if _is_tensor(v):
            t: torch.Tensor = v
            base = f"- {k}: dtype={t.dtype} shape={tuple(t.shape)} numel={t.numel()}"
            if args.stats:
                base += " " + _tensor_stats(t)
            print(base)
            shown += 1
        else:
            if args.show_non_tensors:
                print(f"- {k}: {_summarize_value(v)}")
                shown += 1

        if shown >= args.limit:
            remaining = max(0, len(sd) - shown)
            if remaining:
                print(f"... (stopped at --limit={args.limit}, remaining keys not shown)")
            break


if __name__ == "__main__":
    main()
