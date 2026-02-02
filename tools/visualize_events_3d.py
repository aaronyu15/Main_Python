#!/usr/bin/env python3
"""3D event visualizer for blink_sim-style `events.h5` recordings.

This is aimed at quickly inspecting the raw event stream with x, y, t axes.

Expected HDF5 layout (matches blink_sim generation and Main_Python dataset loader):
- events/x  (u2)
- events/y  (u2)
- events/t  (u8)   timestamps in microseconds
- events/p  (u1)   polarity in {0,1} where 0=negative, 1=positive

Optional (if present) for fast time-range slicing:
- ms_to_idx (u8) mapping millisecond -> event index (monotonic)

Examples
--------
Visualize a whole file with downsampling:
  python blink_sim/scripts/visualize_events_3d.py \
    --events-h5 blink_sim/output/train_set/boy1_BaseballHit_0/events_left/events.h5 \
    --height 320 --width 320 --max-points 200000

Visualize only a time range (fast if ms_to_idx exists):
  python blink_sim/scripts/visualize_events_3d.py \
    --events-h5 .../events.h5 \
    --start-ms 500 --end-ms 650 --max-points 150000

List keys/attrs:
  python blink_sim/scripts/visualize_events_3d.py --events-h5 .../events.h5 --list-keys
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _print_h5_tree(f) -> None:
    def visitor(name, obj):
        kind = "group" if hasattr(obj, "keys") else "dataset"
        print(f"{kind}: {name}")
        try:
            if kind == "dataset":
                print(f"  shape={obj.shape} dtype={obj.dtype}")
        except Exception:
            pass
        try:
            if hasattr(obj, "attrs") and len(obj.attrs) > 0:
                for k in obj.attrs.keys():
                    v = obj.attrs.get(k)
                    vs = v
                    if isinstance(v, (bytes, bytearray)):
                        try:
                            vs = v.decode("utf-8", errors="replace")
                        except Exception:
                            vs = v
                    print(f"  attr {k}: {vs}")
        except Exception:
            pass

    f.visititems(visitor)


def _resolve_events(f):
    required = ["events/x", "events/y", "events/t", "events/p"]
    missing = [k for k in required if k not in f]
    if missing:
        raise KeyError(
            "Missing required datasets in HDF5: "
            + ", ".join(missing)
            + ". Expected blink_sim layout: events/{x,y,t,p}."
        )
    return f["events/x"], f["events/y"], f["events/t"], f["events/p"]


def _slice_indices(
    f,
    t_ds,
    start_ms: Optional[int],
    end_ms: Optional[int],
    start_us: Optional[int],
    end_us: Optional[int],
) -> Tuple[int, int]:
    n = int(t_ds.shape[0])
    if n == 0:
        return 0, 0

    # Prefer ms_to_idx when available: avoids loading all timestamps.
    ms_to_idx = f.get("ms_to_idx", None)
    if ms_to_idx is not None and (start_ms is not None or end_ms is not None or start_us is not None or end_us is not None):
        if start_ms is None and start_us is not None:
            start_ms = int(max(0, start_us // 1000))
        if end_ms is None and end_us is not None:
            end_ms = int(max(0, end_us // 1000))

        ms_len = int(ms_to_idx.shape[0])
        s_ms = 0 if start_ms is None else int(np.clip(start_ms, 0, ms_len - 1))
        e_ms = (ms_len - 1) if end_ms is None else int(np.clip(end_ms, 0, ms_len - 1))
        if e_ms < s_ms:
            e_ms = s_ms

        sidx = int(ms_to_idx[s_ms])
        eidx = int(ms_to_idx[e_ms])
        sidx = int(np.clip(sidx, 0, n))
        eidx = int(np.clip(eidx, 0, n))
        if eidx < sidx:
            eidx = sidx
        return sidx, eidx

    # Fallback: compute indices using timestamps (requires reading timestamps).
    if start_us is None and start_ms is not None:
        start_us = int(start_ms) * 1000
    if end_us is None and end_ms is not None:
        end_us = int(end_ms) * 1000

    if start_us is None and end_us is None:
        return 0, n

    ts = np.asarray(t_ds[:], dtype=np.int64)
    s_us = ts[0] if start_us is None else int(start_us)
    e_us = ts[-1] if end_us is None else int(end_us)
    if e_us < s_us:
        e_us = s_us

    sidx = int(np.searchsorted(ts, s_us, side="left"))
    eidx = int(np.searchsorted(ts, e_us, side="left"))
    sidx = int(np.clip(sidx, 0, n))
    eidx = int(np.clip(eidx, 0, n))
    if eidx < sidx:
        eidx = sidx
    return sidx, eidx


def _downsample_indices(n: int, max_points: int, stride: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)

    stride = max(1, int(stride))
    idx = np.arange(0, n, stride, dtype=np.int64)

    if max_points is not None and max_points > 0 and idx.size > max_points:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(idx, size=int(max_points), replace=False)
        idx.sort()

    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="3D visualize events with x, y, t axes")
    parser.add_argument("--events-h5", type=str, required=True, help="Path to events_left/events.h5")

    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--invert-y", action="store_true", help="Invert Y axis (image-style coordinates)")

    parser.add_argument("--start-ms", type=int, default=None)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--start-us", type=int, default=None)
    parser.add_argument("--end-us", type=int, default=None)

    parser.add_argument("--stride", type=int, default=10, help="Take every Nth event before random subsampling")
    parser.add_argument("--max-points", type=int, default=200_000, help="Max points plotted in 3D scatter")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed used when subsampling")

    parser.add_argument("--point-size", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--t-units", type=str, choices=["us", "ms", "s"], default="ms", help="Display units for the time axis")

    parser.add_argument("--save", type=str, default=None, help="If set, save figure to this path instead of showing")
    parser.add_argument("--list-keys", action="store_true", help="Print HDF5 keys and exit")

    args = parser.parse_args()

    events_path = Path(args.events_h5)
    if not events_path.exists():
        raise FileNotFoundError(str(events_path))

    import h5py

    with h5py.File(events_path, "r") as f:
        if args.list_keys:
            _print_h5_tree(f)
            return

        x_ds, y_ds, t_ds, p_ds = _resolve_events(f)
        sidx, eidx = _slice_indices(f, t_ds, args.start_ms, args.end_ms, args.start_us, args.end_us)
        if eidx <= sidx:
            raise RuntimeError("No events in the requested slice")

        count = eidx - sidx
        idx_local = _downsample_indices(count, args.max_points, args.stride, args.seed)
        idx = (idx_local + sidx).astype(np.int64, copy=False)

        x = np.asarray(x_ds[idx], dtype=np.int32)
        y = np.asarray(y_ds[idx], dtype=np.int32)
        t = np.asarray(t_ds[idx], dtype=np.int64)
        p = np.asarray(p_ds[idx], dtype=np.int8)

    # Polarity: blink_sim saves negative as 0, positive as 1.
    pos = p > 0
    colors = np.zeros((p.shape[0], 4), dtype=np.float32)
    colors[pos] = np.array([1.0, 0.2, 0.2, float(args.alpha)], dtype=np.float32)  # red-ish
    colors[~pos] = np.array([0.2, 0.2, 1.0, float(args.alpha)], dtype=np.float32)  # blue-ish

    if args.invert_y:
        y = (int(args.height) - 1) - y

    # Time axis scaling
    t0 = int(t.min())
    if args.t_units == "us":
        tz = (t - t0).astype(np.float32)
        zlabel = "t (µs from start)"
    elif args.t_units == "ms":
        tz = ((t - t0) / 1e3).astype(np.float32)
        zlabel = "t (ms from start)"
    else:
        tz = ((t - t0) / 1e6).astype(np.float32)
        zlabel = "t (s from start)"

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, tz, s=float(args.point_size), c=colors, marker=".")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_zlabel(zlabel)

    ax.set_xlim(0, int(args.width) - 1)
    ax.set_ylim(0, int(args.height) - 1)

    title = f"{events_path.parent.parent.name}  ({len(x):,} points from {events_path.name})"
    fig.suptitle(title)
    plt.tight_layout()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
