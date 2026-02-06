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


def _load_csv_events(
    csv_path: Path,
    start_ms: Optional[int],
    end_ms: Optional[int],
    start_us: Optional[int],
    end_us: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load events from CSV file with format: x,y,p,t
    
    Args:
        csv_path: Path to CSV file
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        start_us: Start time in microseconds
        end_us: End time in microseconds
    
    Returns:
        Tuple of (x, y, t, p) arrays
    """
    print(f"Loading CSV file: {csv_path}")
    
    # Load CSV data
    data = np.loadtxt(csv_path, delimiter=',', dtype=np.int64)
    
    if data.size == 0:
        raise RuntimeError("CSV file is empty")
    
    # Parse columns: x, y, p, t
    x = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    p = data[:, 2].astype(np.int8)
    t = data[:, 3].astype(np.int64)
    
    print(f"Loaded {len(x):,} events from CSV")
    print(f"Time range: {t.min()} to {t.max()} microseconds")
    
    # Apply time filtering if requested
    if start_us is None and start_ms is not None:
        start_us = int(start_ms) * 1000
    if end_us is None and end_ms is not None:
        end_us = int(end_ms) * 1000
    
    if start_us is not None or end_us is not None:
        s_us = t.min() if start_us is None else int(start_us)
        e_us = t.max() if end_us is None else int(end_us)
        
        if e_us < s_us:
            e_us = s_us
        
        mask = (t >= s_us) & (t <= e_us)
        x = x[mask]
        y = y[mask]
        p = p[mask]
        t = t[mask]
        
        print(f"After time filtering: {len(x):,} events")
    
    if len(x) == 0:
        raise RuntimeError("No events in the requested time slice")
    
    return x, y, t, p


def main() -> None:
    parser = argparse.ArgumentParser(description="3D visualize events with x, y, t axes")
    
    parser.add_argument("--events-file", type=str, required=True, 
                       help="Path to events file (.h5/.hdf5 or .csv format)")

    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--invert-y", action="store_true", help="Invert Y axis (image-style coordinates)")

    parser.add_argument("--start-ms", type=int, default=None)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--start-us", type=int, default=None)
    parser.add_argument("--end-us", type=int, default=None)

    parser.add_argument("--stride", type=int, default=1, help="Take every Nth event before random subsampling")
    parser.add_argument("--max-points", type=int, default=200000, help="Max points plotted in 3D scatter")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed used when subsampling")

    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.00)
    parser.add_argument("--t-units", type=str, choices=["us", "ms", "s"], default="ms", help="Display units for the time axis")

    parser.add_argument("--mode", type=str, choices=["3d", "2d"], default="3d", help="Visualization mode: 3D scatter or 2D frame sequence")
    parser.add_argument("--frame-delta-ms", type=float, default=10.0, help="Time delta in ms between frames for 2D mode")

    parser.add_argument("--save", type=str, default=None, help="If set, save figure to this path instead of showing")
    parser.add_argument("--list-keys", action="store_true", help="Print HDF5 keys and exit (HDF5 only)")

    args = parser.parse_args()

    # Determine input file type and path
    events_path = Path(args.events_file)
    
    if not events_path.exists():
        raise FileNotFoundError(str(events_path))
    
    # Auto-detect file type based on extension
    file_ext = events_path.suffix.lower()
    if file_ext in ['.h5', '.hdf5']:
        file_type = "h5"
        print(f"Detected HDF5 file format")
    elif file_ext == '.csv':
        file_type = "csv"
        print(f"Detected CSV file format")
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}. Use .h5, .hdf5, or .csv")

    # Load events based on file type
    if file_type == "h5":
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
            print(f"Loaded {x.shape[0]:,} events from {events_path} (indices {sidx} to {eidx})")
    else:  # CSV
        if args.list_keys:
            print("--list-keys option is only available for HDF5 files")
            return
        
        x, y, t, p = _load_csv_events(events_path, args.start_ms, args.end_ms, args.start_us, args.end_us)
        
        # Apply stride and downsampling
        count = len(x)
        idx = _downsample_indices(count, args.max_points, args.stride, args.seed)
        
        x = x[idx]
        y = y[idx]
        t = t[idx]
        p = p[idx]
        print(f"After downsampling: {x.shape[0]:,} events")

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

    if args.mode == "3d":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(x, y, tz, s=float(args.point_size), c=colors, marker=".")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_zlabel(zlabel)

        ax.set_xlim(0, int(args.width) - 1)
        ax.set_ylim(0, int(args.height) - 1)

        # Generate title based on file type
        if file_type == "h5":
            title = f"{events_path.parent.parent.name}  ({len(x):,} points from {events_path.name})"
        else:
            title = f"{events_path.stem}  ({len(x):,} points from {events_path.name})"
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
    else:
        # 2D frame-based visualization
        from matplotlib.widgets import Slider
        
        # Convert frame delta to microseconds
        frame_delta_us = int(args.frame_delta_ms * 1000)
        
        # Create time bins for frames
        t_min = int(t.min())
        t_max = int(t.max())
        t_range = t_max - t_min
        
        if t_range < frame_delta_us:
            frame_delta_us = max(1, t_range)
        
        num_frames = int(np.ceil(t_range / frame_delta_us))
        print(f"Creating {num_frames} frames with {args.frame_delta_ms:.2f} ms delta")
        
        # Precompute frames
        frames = []
        for i in range(num_frames):
            t_start = t_min + i * frame_delta_us
            t_end = t_start + frame_delta_us
            mask = (t >= t_start) & (t < t_end)
            
            frame_x = x[mask]
            frame_y = y[mask]
            frame_p = p[mask]
            frames.append((frame_x, frame_y, frame_p, t_start, t_end))
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 9))
        plt.subplots_adjust(bottom=0.15)
        
        # Initial frame
        current_frame = [0]
        
        def plot_frame(frame_idx):
            ax.clear()
            frame_x, frame_y, frame_p, t_start, t_end = frames[frame_idx]
            
            # Plot positive events (red) and negative events (blue)
            pos_mask = frame_p > 0
            if np.any(pos_mask):
                ax.scatter(frame_x[pos_mask], frame_y[pos_mask], 
                          s=float(args.point_size), c='red', marker='.', alpha=float(args.alpha), label='Positive')
            if np.any(~pos_mask):
                ax.scatter(frame_x[~pos_mask], frame_y[~pos_mask], 
                          s=float(args.point_size), c='blue', marker='.', alpha=float(args.alpha), label='Negative')
            
            ax.set_xlim(0, int(args.width) - 1)
            ax.set_ylim(0, int(args.height) - 1)
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            # Calculate time in appropriate units
            if args.t_units == "us":
                t_display_start = (t_start - t_min)
                t_display_end = (t_end - t_min)
                time_label = f"t: {t_display_start:.0f}-{t_display_end:.0f} µs"
            elif args.t_units == "ms":
                t_display_start = (t_start - t_min) / 1e3
                t_display_end = (t_end - t_min) / 1e3
                time_label = f"t: {t_display_start:.2f}-{t_display_end:.2f} ms"
            else:
                t_display_start = (t_start - t_min) / 1e6
                t_display_end = (t_end - t_min) / 1e6
                time_label = f"t: {t_display_start:.3f}-{t_display_end:.3f} s"
            
            # Generate title based on file type
            if file_type == "h5":
                title = f"{events_path.parent.parent.name} - Frame {frame_idx + 1}/{num_frames}\n{time_label} ({len(frame_x):,} events)"
            else:
                title = f"{events_path.stem} - Frame {frame_idx + 1}/{num_frames}\n{time_label} ({len(frame_x):,} events)"
            ax.set_title(title)
            if np.any(pos_mask) or np.any(~pos_mask):
                ax.legend(loc='upper right')
            fig.canvas.draw_idle()
        
        # Create slider
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
        
        def update(val):
            frame_idx = int(slider.val)
            current_frame[0] = frame_idx
            plot_frame(frame_idx)
        
        slider.on_changed(update)
        
        # Plot initial frame
        plot_frame(0)
        
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
