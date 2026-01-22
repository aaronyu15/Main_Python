#!/usr/bin/env python3

"""Visualize weight distributions from a PyTorch checkpoint.

This script loads a checkpoint (.pth), selects a state_dict, and writes one
histogram plot per layer/parameter tensor (typically weights and biases).

Examples:
  python tools/visualize_checkpoint_weights.py checkpoints/full_p/best_model.pth
  python tools/visualize_checkpoint_weights.py checkpoints/full_p/best_model.pth --grep "e[123]\\.conv" --stats
  python tools/visualize_checkpoint_weights.py checkpoints/full_p/best_model.pth --out-dir ./output/weight_hists --logy

Notes:
- Uses matplotlib; works headless (writes PNGs).
- By default, plots tensors whose key ends with ".weight" or ".bias".
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


DEFAULT_STATE_KEYS = ("model_state_dict", "state_dict", "model")


def _load_checkpoint(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


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

        if all(isinstance(kk, str) for kk in ckpt.keys()):
            return "<root>", ckpt  # type: ignore[return-value]

    return None, None


def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:200]


@torch.no_grad()
def _tensor_to_1d_numpy(t: torch.Tensor) -> np.ndarray:
    tt = t.detach().cpu()
    if tt.is_sparse:
        tt = tt.to_dense()
    # float64 for stable stats/plot
    return tt.flatten().double().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot per-layer weight histograms from a checkpoint")
    ap.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    ap.add_argument(
        "--state-dict",
        type=str,
        default=None,
        help="Force which top-level key contains the state_dict (e.g. model_state_dict)",
    )
    ap.add_argument(
        "--grep",
        type=str,
        default=r"\.(weight|bias)$",
        help="Regex filter applied to parameter keys",
    )
    ap.add_argument("--bins", type=int, default=200, help="Histogram bins")
    ap.add_argument("--logy", action="store_true", help="Log-scale the y-axis")
    ap.add_argument("--stats", action="store_true", help="Annotate plots with mean/std/min/max")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "weight_hists"),
        help="Directory to write PNGs",
    )
    ap.add_argument("--max-plots", type=int, default=1000, help="Safety cap on number of plots")
    ap.add_argument("--dpi", type=int, default=150, help="PNG DPI")
    ap.add_argument("--show", action="store_true", help="Also show plots interactively")

    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _load_checkpoint(ckpt_path)
    state_key, sd = _choose_state_dict(ckpt, args.state_dict)
    if sd is None:
        raise SystemExit(
            "Could not find a state_dict in the checkpoint. "
            f"Tried keys: {', '.join(DEFAULT_STATE_KEYS)}"
        )

    key_re = re.compile(args.grep) if args.grep else None

    # Import matplotlib lazily so environments without display still work.
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    keys = sorted(k for k in sd.keys() if isinstance(k, str))

    selected = []
    for k in keys:
        if key_re is not None and key_re.search(k) is None:
            continue
        v = sd[k]
        if not isinstance(v, torch.Tensor):
            continue
        selected.append(k)

    print(f"[weights] checkpoint: {ckpt_path}")
    print(f"[weights] state_dict: {state_key}")
    print(f"[weights] writing to: {out_dir}")
    print(f"[weights] selected tensors: {len(selected)}")

    if len(selected) == 0:
        print("[weights] No tensors matched filters. Try relaxing --include/--grep.")
        return

    if len(selected) > args.max_plots:
        print(f"[weights] Refusing to write {len(selected)} plots (cap={args.max_plots}).")
        print("[weights] Use --max-plots to override.")
        return

    for k in selected:
        t = sd[k]
        assert isinstance(t, torch.Tensor)

        x = _tensor_to_1d_numpy(t)
        x = x[np.isfinite(x)]
        if x.size == 0:
            print(f"[skip] {k}: all values are non-finite")
            continue

        fig = plt.figure(figsize=(7.5, 4.2))
        ax = fig.add_subplot(1, 1, 1)

        ax.hist(x, bins=args.bins, color="#2b6cb0", alpha=0.85)
        ax.set_title(k)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        if args.logy:
            ax.set_yscale("log")

        if args.stats:
            mean = float(np.mean(x))
            std = float(np.std(x))
            vmin = float(np.min(x))
            vmax = float(np.max(x))
            txt = f"dtype={t.dtype} shape={tuple(t.shape)}\nmean={mean:.4g} std={std:.4g}\nmin={vmin:.4g} max={vmax:.4g}"
            ax.text(
                0.98,
                0.98,
                txt,
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=9,
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
            )

        fig.tight_layout()

        fname = _sanitize_filename(k) + ".png"
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"[write] {out_path}")

    if args.show:
        # If interactive show was requested, re-open the last figure window.
        # (Typical use is headless save; this is a convenience.)
        plt.show()


if __name__ == "__main__":
    main()
