"""
DSEC Optical Flow Dataset Loader
Drop-in replacement for OpticalFlowDataset that loads from the DSEC dataset structure.

DSEC flat directory structure:
    dsec_root/            (e.g. dsec/train/ or dsec/test/)
        sequence_name/
            events.h5                                       (events/x, events/y, events/t, events/p)
            rectify_map.h5                                  (rectify_map [H, W, 2])
            <seq>_optical_flow_forward_timestamps.txt        (train: from_ts, to_ts)
            <seq>.csv                                       (test:  from_ts, to_ts, file_index)
            000134.png, 000136.png, ...                     (train only: 16-bit PNG flow GT)

Train split:  has flow PNGs + timestamps .txt    -> supervised training / evaluation
Test split:   has CSV + no flow PNGs             -> inference only (flow & valid_mask are zeros)

Flow PNG encoding (DSEC spec):
    Channel 0 (R) = horizontal flow u, uint16, flow_u = (value - 2^15) / 128.0
    Channel 1 (G) = vertical flow v,   uint16, flow_v = (value - 2^15) / 128.0
    Channel 2 (B) = valid mask,         uint16, valid  = (value > 0)

Rectification:
    DSEC events are stored in raw (distorted) sensor coordinates, but the flow
    ground truth is in rectified image space.  The rectify_map.h5 file provides
    a per-pixel lookup [H, W, 2] mapping raw (x, y) -> rectified (x', y').
    Events are remapped with bilinear splatting before building the voxel grid
    so that they align with the flow.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import hdf5plugin  # registers HDF5 compression filters (blosc, lz4, etc.)
from typing import Dict, List, Tuple, Optional


class DSECOpticalFlowDataset(Dataset):
    """
    DSEC optical flow dataset loader.

    Accepts the same config dict as OpticalFlowDataset so it can be used as a
    drop-in replacement in evaluate.py.  Only config keys that are relevant to
    DSEC are used; training-only keys (patch_mode, augment_rotation, etc.) are
    silently ignored.
    """

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}

        self.dsec_root = Path(config.get('data_root', '../dsec'))
        self.num_bins = config.get('num_bins', 5)
        self.bin_interval_us = config.get('bin_interval_us', 10000)
        self.use_polarity = config.get('use_polarity', False)
        self.data_size: Tuple[int, int] = tuple(config.get('data_size', (480, 640)))  # (H, W)
        self.max_samples = config.get('max_train_samples', None)

        # DSEC native resolution
        self.dsec_height = 480
        self.dsec_width = 640

        # Build sample list
        self.samples = self._build_sample_list()

        # Whether this split has ground-truth flow
        self.has_flow = len(self.samples) > 0 and self.samples[0]['flow_path'] is not None

        # Per-sequence caches (lightweight — only ms_to_idx + t_offset + rectify)
        self._seq_cache: Dict[str, Dict] = {}
        self._rectify_cache: Dict[str, np.ndarray] = {}
        self._preload_sequences()

    # ------------------------------------------------------------------
    # Sample discovery
    # ------------------------------------------------------------------
    def _build_sample_list(self) -> List[Dict]:
        samples = []
        root = self.dsec_root

        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir():
                continue
            seq_name = seq_dir.name

            # Locate events.h5 (flat layout)
            event_h5 = seq_dir / 'events.h5'
            if not event_h5.exists():
                continue

            # Locate rectification map
            rectify_h5 = seq_dir / 'rectify_map.h5'
            if not rectify_h5.exists():
                rectify_h5 = None

            # Determine split type: train has *_timestamps.txt, test has <seq>.csv
            ts_file = seq_dir / f'{seq_name}_optical_flow_forward_timestamps.txt'
            csv_file = seq_dir / f'{seq_name}.csv'

            if ts_file.exists():
                # ---- Train split: timestamps + flow PNGs ----
                flow_pngs = sorted(seq_dir.glob('*.png'))

                with open(ts_file, 'r') as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

                if len(lines) != len(flow_pngs):
                    print(f"WARNING: {seq_name}: {len(lines)} timestamp lines "
                          f"vs {len(flow_pngs)} PNGs — skipping")
                    continue

                for i, (line, png_path) in enumerate(zip(lines, flow_pngs)):
                    parts = line.split(',')
                    if len(parts) < 2:
                        continue
                    from_ts, to_ts = int(parts[0].strip()), int(parts[1].strip())
                    samples.append({
                        'sequence': seq_name,
                        'flow_path': png_path,
                        'event_h5_path': event_h5,
                        'rectify_h5_path': rectify_h5,
                        'from_ts': from_ts,
                        'to_ts': to_ts,
                        'index': i,
                        'file_index': -1,
                    })

            elif csv_file.exists():
                # ---- Test split: CSV with file_index, no flow PNGs ----
                with open(csv_file, 'r') as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

                for i, line in enumerate(lines):
                    parts = line.split(',')
                    if len(parts) < 3:
                        continue
                    from_ts = int(parts[0].strip())
                    to_ts = int(parts[1].strip())
                    file_index = int(parts[2].strip())
                    samples.append({
                        'sequence': seq_name,
                        'flow_path': None,
                        'event_h5_path': event_h5,
                        'rectify_h5_path': rectify_h5,
                        'from_ts': from_ts,
                        'to_ts': to_ts,
                        'index': i,
                        'file_index': file_index,
                    })
            else:
                # No timestamp/csv file — skip sequence
                continue

        # Optionally limit number of samples
        if self.max_samples is not None and self.max_samples < len(samples):
            samples = samples[:self.max_samples]

        return samples

    # ------------------------------------------------------------------
    # Per-sequence caching (lightweight — mirrors MVSEC approach)
    # ------------------------------------------------------------------
    def _preload_sequences(self):
        """Cache only the tiny ms_to_idx + t_offset per sequence.

        DSEC events.h5 contains a ``ms_to_idx`` dataset that maps each
        millisecond offset from the recording start to the corresponding
        event index.  This is only ~100 KB per sequence (vs. tens of GB
        for the full timestamp array).

        In __getitem__ we use ms_to_idx for a coarse bracket, read a
        small HDF5 slice, then do a fine binary search — identical to the
        strategy used in the MVSEC loader.
        """
        seen: set = set()
        for s in self.samples:
            key = str(s['event_h5_path'])
            if key in seen:
                continue
            seen.add(key)

            with h5py.File(s['event_h5_path'], 'r') as f:
                ms_to_idx = f['ms_to_idx'][:].astype(np.int64)
                if 't_offset' in f:
                    t_offset = int(f['t_offset'][()])
                elif 'events/t_offset' in f:
                    t_offset = int(f['events/t_offset'][()])
                else:
                    t_offset = 0
                num_events = f['events/t'].shape[0]

            self._seq_cache[key] = {
                'ms_to_idx': ms_to_idx,
                't_offset': t_offset,
                'num_events': num_events,
            }

        total_kb = sum(c['ms_to_idx'].nbytes for c in self._seq_cache.values()) / 1024
        print(f"DSECOpticalFlowDataset: cached ms_to_idx for "
              f"{len(self._seq_cache)} sequence(s) ({total_kb:.0f} KB total)")

    def _find_event_range(self, h5_path: Path, t0_us: int, t1_us: int) -> Tuple[int, int]:
        """Two-step event lookup using ms_to_idx (coarse) + HDF5 slice (fine).

        Args:
            h5_path: Path to the events.h5 file.
            t0_us, t1_us: Absolute timestamps in microseconds defining the
                event window [t0_us, t1_us).

        Returns:
            (start_idx, end_idx) into the events arrays.
        """
        cache = self._seq_cache[str(h5_path)]
        ms_to_idx = cache['ms_to_idx']
        t_offset = cache['t_offset']
        num_events = cache['num_events']

        # Convert absolute timestamps to raw (relative to recording start)
        t0_raw = t0_us - t_offset
        t1_raw = t1_us - t_offset

        # Convert to millisecond indices (with 1 ms margin)
        ms0 = max(0, int(t0_raw // 1000) - 1)
        ms1 = min(len(ms_to_idx) - 1, int(t1_raw // 1000) + 1)

        coarse_start = int(ms_to_idx[ms0])
        coarse_end = int(ms_to_idx[ms1]) if ms1 < len(ms_to_idx) else num_events

        # Fine search: read only the timestamps in the coarse range
        with h5py.File(h5_path, 'r') as f:
            chunk_t = f['events/t'][coarse_start:coarse_end].astype(np.int64)

        # Binary search within the chunk (raw timestamps, microseconds)
        fine_start = np.searchsorted(chunk_t, t0_raw, side='left')
        fine_end = np.searchsorted(chunk_t, t1_raw, side='right')

        return coarse_start + fine_start, coarse_start + fine_end

    def _get_rectify_map(self, h5_path: Optional[Path]) -> Optional[np.ndarray]:
        """
        Return the rectification map [H, W, 2] for a sequence, cached.
        Returns None if no rectify_map.h5 was found for this sequence.
        """
        if h5_path is None:
            return None
        key = str(h5_path)
        if key not in self._rectify_cache:
            with h5py.File(h5_path, 'r') as f:
                self._rectify_cache[key] = f['rectify_map'][()].astype(np.float32)
        return self._rectify_cache[key]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import cv2

        sample = self.samples[idx]

        # ---- Load flow ground truth from 16-bit PNG ----
        if sample['flow_path'] is not None:
            # cv2 with IMREAD_UNCHANGED reliably reads uint16 3-channel PNGs.
            # PNG stores RGB; OpenCV returns BGR, so channel indices are swapped:
            #   file channel 0 (u)     -> cv2 channel 2
            #   file channel 1 (v)     -> cv2 channel 1
            #   file channel 2 (valid) -> cv2 channel 0
            flow_png = cv2.imread(str(sample['flow_path']), cv2.IMREAD_UNCHANGED)
            if flow_png is None:
                raise FileNotFoundError(f"Could not read flow PNG: {sample['flow_path']}")

            flow_u = (flow_png[:, :, 2].astype(np.float32) - 2**15) / 128.0
            flow_v = (flow_png[:, :, 1].astype(np.float32) - 2**15) / 128.0
            valid  = (flow_png[:, :, 0] > 0).astype(np.float32)

            flow = np.stack([flow_u, flow_v], axis=0)       # [2, H, W]
            valid_mask = valid[np.newaxis, :, :]             # [1, H, W]
        else:
            # Test split: no GT flow available
            flow = np.zeros((2, self.dsec_height, self.dsec_width), dtype=np.float32)
            valid_mask = np.zeros((1, self.dsec_height, self.dsec_width), dtype=np.float32)

        # ---- Load events for this time window ----
        # Match the training dataset's windowing: anchor at to_ts, look back
        # by num_bins * bin_interval_us so the voxel grid temporal coverage is
        # identical to what the model was trained on.
        time_window_us = self.num_bins * self.bin_interval_us
        t1 = sample['to_ts']
        t0 = t1 - time_window_us

        # Two-step lookup: coarse bracket via ms_to_idx, fine via HDF5 slice
        start_idx, end_idx = self._find_event_range(sample['event_h5_path'], t0, t1)

        with h5py.File(sample['event_h5_path'], 'r') as f:
            x = f['events/x'][start_idx:end_idx].astype(np.int32)
            y = f['events/y'][start_idx:end_idx].astype(np.int32)
            t = f['events/t'][start_idx:end_idx].astype(np.float64)
            p = f['events/p'][start_idx:end_idx].astype(np.int8)

        # DSEC stores polarity as 0/1 -> remap to -1/+1
        p = np.where(p == 0, np.int8(-1), np.int8(1))

        # ---- Rectify event coordinates ----
        # Flow GT is in rectified space; raw events are distorted.
        # The rectify_map gives the rectified (x', y') for each raw pixel.
        rect_map = self._get_rectify_map(sample['rectify_h5_path'])
        if rect_map is not None:
            # Clamp raw coords to valid lookup range
            x_clamped = np.clip(x, 0, rect_map.shape[1] - 1)
            y_clamped = np.clip(y, 0, rect_map.shape[0] - 1)
            # Look up rectified floating-point coordinates
            x_rect = rect_map[y_clamped, x_clamped, 0]
            y_rect = rect_map[y_clamped, x_clamped, 1]
        else:
            x_rect = x.astype(np.float32)
            y_rect = y.astype(np.float32)

        # ---- Build voxel grid at DSEC native resolution ----
        voxel = self._events_to_voxel_grid(x_rect, y_rect, t, p,
                                           self.dsec_height, self.dsec_width)

        # ---- Spatial alignment: crop / resize to data_size ----
        target_h, target_w = self.data_size
        src_h, src_w = self.dsec_height, self.dsec_width

        input_tensor = torch.from_numpy(voxel)
        flow_tensor  = torch.from_numpy(flow)
        valid_tensor = torch.from_numpy(valid_mask)

        return {
            'input': input_tensor,          # [num_bins, C, H, W]
            'flow': flow_tensor,            # [2, H, W]
            'valid_mask': valid_tensor,      # [1, H, W]
            'metadata': {
                'sequence': sample['sequence'],
                'index': sample['index'],
                'file_index': sample['file_index'],
            }
        }

    # ------------------------------------------------------------------
    # Voxel grid construction (vectorised, with bilinear splatting)
    # ------------------------------------------------------------------
    def _events_to_voxel_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Vectorised voxel grid construction with bilinear splatting.

        Event coordinates (x, y) may be floating-point (after rectification).
        Each event is distributed across its 4 nearest integer pixel neighbours
        with weights proportional to the overlap area, preserving the total
        event count.

        Returns:
            np.ndarray of shape [num_bins, C, H, W]
            where C=2 if use_polarity else C=1.
        """
        n_chan = 2 if self.use_polarity else 1
        voxel = np.zeros((self.num_bins, n_chan, height, width), dtype=np.float32)

        if len(x) == 0:
            return voxel

        # Normalise timestamps into bin indices [0, num_bins)
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = ((t - t_min) / (t_max - t_min) * (self.num_bins - 1e-6))
        else:
            t_norm = np.zeros_like(t, dtype=np.float64)

        bin_idx = t_norm.astype(np.int32)

        # Polarity channel index
        if self.use_polarity:
            pol_idx = np.where(p > 0, 0, 1).astype(np.int32)
        else:
            pol_idx = np.zeros(len(x), dtype=np.int32)

        # ---- Bilinear splatting ----
        # Floor coordinates and fractional remainders
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        dx = (x - x0).astype(np.float32)
        dy = (y - y0).astype(np.float32)

        # Four corner coordinates
        x1 = x0 + 1
        y1 = y0 + 1

        # Bilinear weights (area of opposite corner)
        w00 = (1.0 - dx) * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w10 = dx         * (1.0 - dy)
        w11 = dx         * dy

        # Splat each corner, masking out-of-bounds pixels
        for (cx, cy, w) in [(x0, y0, w00), (x0, y1, w01),
                            (x1, y0, w10), (x1, y1, w11)]:
            mask = (
                (bin_idx >= 0) & (bin_idx < self.num_bins) &
                (cx >= 0) & (cx < width) &
                (cy >= 0) & (cy < height)
            )
            np.add.at(voxel, (bin_idx[mask], pol_idx[mask],
                              cy[mask], cx[mask]), w[mask])

        # Binarize: any pixel that received any event contribution -> 1.0
        voxel[voxel > 0] = 1.0

        return voxel


# ======================================================================
# Debug visualisation
# ======================================================================
if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import random

    parser = argparse.ArgumentParser(description='Debug: visualise a DSEC sample')
    parser.add_argument('--dsec-root', type=str, default='../dsec',
                        help='Path to DSEC dataset root')
    parser.add_argument('--num-bins', type=int, default=5)
    parser.add_argument('--bin-interval-us', type=int, default=10000)
    parser.add_argument('--use-polarity', action='store_true')
    parser.add_argument('--data-size', type=int, nargs=2, default=[480, 640],
                        help='H W target size')
    parser.add_argument('--index', type=int, default=None,
                        help='Sample index (random if not set)')
    args = parser.parse_args()

    config = {
        'data_root': args.dsec_root,
        'num_bins': args.num_bins,
        'bin_interval_us': args.bin_interval_us,
        'use_polarity': args.use_polarity,
        'data_size': tuple(args.data_size),
    }

    print(f"Loading DSEC dataset from {args.dsec_root} ...")
    ds = DSECOpticalFlowDataset(config=config)
    print(f"Found {len(ds)} samples across "
          f"{len(set(s['sequence'] for s in ds.samples))} sequence(s)")

    if len(ds) == 0:
        print("No samples found -- check your dsec_root path and directory layout.")
        raise SystemExit(1)

    idx = args.index if args.index is not None else random.randint(0, len(ds) - 1)
    print(f"Loading sample {idx} ...")

    # ---- Timestamp diagnostics ----
    s = ds.samples[idx]
    time_window = ds.num_bins * ds.bin_interval_us
    t1 = s['to_ts']
    t0 = t1 - time_window
    start_idx, end_idx = ds._find_event_range(s['event_h5_path'], t0, t1)

    cache = ds._seq_cache[str(s['event_h5_path'])]
    print(f"\n  --- Timestamp diagnostics ---")
    print(f"  t_offset       : {cache['t_offset']}")
    print(f"  num_events     : {cache['num_events']}")
    print(f"  ms_to_idx len  : {len(cache['ms_to_idx'])}")
    print(f"  from_ts, to_ts : {s['from_ts']}, {s['to_ts']}")
    print(f"  to_ts - from_ts: {s['to_ts'] - s['from_ts']}")
    print(f"  bin_interval_us: {ds.bin_interval_us}")
    print(f"  time_window    : {time_window}  (num_bins={ds.num_bins} x bin_interval={ds.bin_interval_us})")
    print(f"  Computed [t0,t1]: [{t0}, {t1}]")
    print(f"  event range    : start_idx={start_idx}, end_idx={end_idx}, n_events={end_idx - start_idx}")
    print(f"  Rectify map    : {s['rectify_h5_path'] or 'NOT FOUND'}")
    print()

    sample = ds[idx]

    inp = sample['input']           # [T, C, H, W]
    flow = sample['flow']           # [2, H, W]
    valid = sample['valid_mask']    # [1, H, W]
    meta = sample['metadata']

    print(f"  Sequence : {meta['sequence']}")
    print(f"  Index    : {meta['index']}")
    print(f"  file_idx : {meta['file_index']}")
    print(f"  Input    : {inp.shape}  (T={inp.shape[0]}, C={inp.shape[1]})")
    print(f"  Flow     : {flow.shape}  u=[{flow[0].min():.2f}, {flow[0].max():.2f}]  "
          f"v=[{flow[1].min():.2f}, {flow[1].max():.2f}]")
    print(f"  Valid    : {valid.shape}  ({valid.sum().long().item()} valid pixels)")
    print(f"  Events   : min={inp.min():.0f}  max={inp.max():.0f}  "
          f"total={inp.sum():.0f}")
    print(f"  has_flow : {ds.has_flow}")

    # ---- Helper: flow -> RGB (HSV wheel) ----
    def flow_to_rgb(flow_np, mask=None):
        """flow_np: [H, W, 2], mask: [H, W] bool/float (optional)"""
        mag = np.linalg.norm(flow_np, axis=2)
        ang = np.arctan2(flow_np[..., 1], flow_np[..., 0])
        hsv = np.zeros((*flow_np.shape[:2], 3), dtype=np.float32)
        hsv[..., 0] = (ang + np.pi) / (2 * np.pi)   # hue
        hsv[..., 1] = 1.0                            # saturation
        # Normalise by 95th percentile so outliers don't crush the rest
        mag_ref = np.percentile(mag[mag > 0], 95) if (mag > 0).any() else 1.0
        hsv[..., 2] = np.clip(mag / (mag_ref + 1e-6), 0, 1)  # value
        rgb = mcolors.hsv_to_rgb(hsv)
        # Set invalid / zero-flow regions to white
        if mask is not None:
            rgb[mask < 0.5] = 1.0
        return rgb

    # ---- Build figure ----
    n_bins = inp.shape[0]
    n_cols = n_bins + (2 if ds.has_flow else 0)  # bins + (GT flow + valid mask) if train
    fig, axes = plt.subplots(1, max(n_cols, 1), figsize=(3.2 * max(n_cols, 1), 3.5))
    if n_cols == 1:
        axes = [axes]

    # Per-bin event images
    for b in range(n_bins):
        ax = axes[b]
        frame = inp[b].sum(dim=0).numpy()     # sum over polarity channels -> [H, W]
        ax.imshow(frame, cmap='Greys', interpolation='nearest')
        ax.set_title(f'Bin {b}', fontsize=9)
        ax.axis('off')

    if ds.has_flow:
        # GT flow
        ax_flow = axes[n_bins]
        flow_hw2 = flow.permute(1, 2, 0).numpy()  # [H, W, 2]
        ax_flow.imshow(flow_to_rgb(flow_hw2, mask=valid[0].numpy()))
        ax_flow.set_title('GT Flow', fontsize=9)
        ax_flow.axis('off')

        # Valid mask
        ax_mask = axes[n_bins + 1]
        ax_mask.imshow(valid[0].numpy(), cmap='Greys', vmin=0, vmax=1)
        ax_mask.set_title('Valid Mask', fontsize=9)
        ax_mask.axis('off')

    fig.suptitle(f"{meta['sequence']}  idx={meta['index']}  "
                 f"(sample {idx}/{len(ds)})", fontsize=11)
    plt.tight_layout()
    out_path = f'dsec_debug_sample_{idx}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved visualisation to {out_path}")