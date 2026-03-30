"""
MVSEC Optical Flow Dataset Loader

Loads event data and ground truth optical flow from the MVSEC dataset
(Multi Vehicle Stereo Event Camera Dataset).

Expected files in data_root:
    indoor_flying1_data.hdf5
    indoor_flying1_gt_flow_dist.npz
    indoor_flying2_data.hdf5
    indoor_flying2_gt_flow_dist.npz
    ...
    outdoor_day1_data.hdf5
    outdoor_day1_gt_flow_dist.npz
    ...

HDF5 structure (data files):
    davis/left/events          - [N, 4] array: (x, y, timestamp_s, polarity)
    davis/left/image_raw       - grayscale frames
    davis/left/image_raw_ts    - frame timestamps (seconds)

NPZ structure (flow files):
    timestamps    - [num_frames] GT flow timestamps (seconds)
    x_flow_dist   - [num_frames, 260, 346] horizontal flow (pixels)
    y_flow_dist   - [num_frames, 260, 346] vertical flow (pixels)

Compatible with OpticalFlowDataset interface — returns the same dict format.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import h5py


# DAVIS346 sensor resolution
MVSEC_HEIGHT = 260
MVSEC_WIDTH = 346

# For outdoor sequences the bottom rows show the car hood and should be masked.
# E-RAFT (Gehrig et al., 3DV 2021) masks rows >= 193 in loader_mvsec_flow.py.
# Brebion et al. (IEEE T-ITS 2022) confirm this is standard practice.
OUTDOOR_HOOD_ROW = 193

# Frame index ranges for MVSEC evaluation.
#
# IMPORTANT: There is no single authoritative source for these ranges.
# No published paper explicitly states the frame indices used; ranges
# live entirely in code configs and are passed between implementations.
#
# What we can verify from public sources:
#   - outdoor_day1 test: E-RAFT uses range(4356, 4706) in its mvsec_20.json
#     config (~350 frames at 20Hz ≈ 18s). Brebion et al. (IEEE T-ITS 2022)
#     note that "the authors of EV-FlowNet only evaluated [outdoor_day1]
#     on a carefully selected 18-second-long extract."
#   - outdoor_day2: used for training by EV-FlowNet, Spike-FlowNet, and
#     most subsequent work. No published range restrictions.
#   - indoor_flying 1/2/3: used for evaluation. No verifiable frame ranges
#     found in any paper or public config.
#
# The defaults below use the E-RAFT outdoor_day1 test range as the only
# concrete number we have.  For all other sequences, no range is applied;
# invalid GT frames are handled by the per-pixel NaN/zero validity mask.
#
# You can (and should) override these via the 'filter' config key.
DEFAULT_FILTER = {
    'outdoor_day1': (4356, 4706),  # E-RAFT mvsec_20.json test split
}


class MVSECDataset(Dataset):
    """
    Dataset for loading optical flow data from MVSEC.

    Produces samples identical in structure to OpticalFlowDataset:
        'input'      : [num_bins, C, H, W]  event voxel grid
        'flow'       : [2, H, W]            (u, v) optical flow in pixels
        'valid_mask' : [1, H, W]            binary validity mask
        'metadata'   : dict                 sequence name, index, dt, etc.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Config keys (with defaults):
            data_root          : str   – directory containing the HDF5/NPZ files
            sequences          : list  – which sequences to load, e.g.
                                         ['indoor_flying1', 'outdoor_day1']
                                         Default: all sequences found in data_root
            dt                 : int   – temporal stride for flow accumulation
                                         1 = consecutive GT frames (~20 Hz)
                                         4 = 4-frame accumulation
            num_bins           : int   – number of temporal bins for voxel grid (default 5)
            bin_interval_us    : int   – duration of each temporal bin in microseconds (default 10000)
                                         Event window = num_bins * bin_interval_us, anchored at
                                         the flow destination timestamp and extending backwards.
            use_polarity       : bool  – separate polarity channels (default False)
            mask_outdoor_hood  : bool  – zero out car-hood region in outdoor seqs (default True)
            filter             : dict  – per-sequence (start, end) frame ranges, e.g.
                                         {'outdoor_day1': (4356, 4706)}
                                         Sequences not in filter use all available frames.
                                         Set to {} to disable all range restrictions.
                                         Default: DEFAULT_FILTER (outdoor_day1 E-RAFT test split)
            flow_clip_range    : tuple – optional (min, max) to clamp flow values
            crop_size          : tuple – optional (H, W) center crop applied to all outputs.
                                         E.g. (260, 340) to trim 3px from each side of the
                                         346-wide DAVIS sensor.  Default: None (no crop).
            augment_rotation   : bool  – random 90° rotation augmentation (default False)
            patch_mode         : bool  – extract patches (default False)
            patch_size         : int   – patch size when patch_mode=True (default 64)
            activity_threshold : int   – min events for patch candidate (default 5)
        """
        if config is None:
            config = {}

        self.data_root = Path(config.get('data_root', './mvsec'))
        self.dt = config.get('dt', 1)
        self.num_bins = config.get('num_bins', 5)
        self.bin_interval_us = config.get('bin_interval_us', 10000)
        self.use_polarity = config.get('use_polarity', False)
        self.data_size = (MVSEC_HEIGHT, MVSEC_WIDTH)

        self.mask_outdoor_hood = config.get('mask_outdoor_hood', True)
        self.filter = config.get('filter', DEFAULT_FILTER)
        self.flow_clip_range = config.get('flow_clip_range', None)
        self.crop_size = config.get('crop_size', None)  # (H, W) or None

        self.augment_rotation = config.get('augment_rotation', False)
        self.patch_mode = config.get('patch_mode', False)
        self.patch_size = config.get('patch_size', 64)
        self.activity_threshold = config.get('activity_threshold', 5)
        self.return_full_frame = config.get('return_full_frame', False)

        self.flip_left_to_right_prob = config.get('flip_left_to_right_prob', 0.0)
        self.flip_left_threshold = config.get('flip_left_threshold', -0.05)

        # Discover or filter sequences
        requested = config.get('sequences', None)
        self.sequence_names = self._find_sequences(requested)

        # Pre-load data that would be expensive to reload on every __getitem__.
        # This trades memory for speed — a few hundred MB per sequence.
        self._cache = {}  # keyed by sequence name
        self._preload_sequences()

        # Build sample list (needs _cache to be populated for frame counts)
        self.samples = self._build_sample_list()

    def _preload_sequences(self):
        """Memory-map GT flow and build a lightweight event index.

        GT flow arrays are memory-mapped (mmap_mode='r') so the OS only
        pages in the frames actually accessed.

        Instead of loading all ~30M event timestamps into RAM (~240MB per
        sequence), we cache the MVSEC-provided image_raw_ts and
        image_raw_event_inds arrays (~few KB each).  These map each
        grayscale frame to its nearest event index, giving us a coarse
        lookup table.  In __getitem__ we use this to narrow the search
        to a small HDF5 slice, then do a precise binary search within it.

        Only numpy arrays / mmap references are cached (safe to fork with
        DataLoader workers). HDF5 file handles are NOT kept open.
        """
        for seq_name in self.sequence_names:
            data_path = self.data_root / f'{seq_name}_data.hdf5'
            flow_path = self.data_root / f'{seq_name}_gt_flow_dist.npz'

            # Memory-map GT flow — arrays stay on disk, paged in on demand.
            gt = np.load(str(flow_path), mmap_mode='r')

            # Load the coarse event index from the HDF5.
            # image_raw_ts:         [N_images]  timestamps of grayscale frames
            # image_raw_event_inds: [N_images]  event index nearest to each frame
            with h5py.File(str(data_path), 'r') as h5:
                img_ts = h5['davis']['left']['image_raw_ts'][:].astype(np.float64)
                img_ev_inds = h5['davis']['left']['image_raw_event_inds'][:].astype(np.int64)
                num_events = h5['davis']['left']['events'].shape[0]

            self._cache[seq_name] = {
                'gt_timestamps': gt['timestamps'],      # tiny, ~35KB
                'x_flow_dist':   gt['x_flow_dist'],     # mmap
                'y_flow_dist':   gt['y_flow_dist'],     # mmap
                'img_ts':        img_ts,                 # ~few KB
                'img_ev_inds':   img_ev_inds,            # ~few KB
                'num_events':    num_events,
                'data_path':     str(data_path),
            }

        print(f"MVSECDataset: preloaded {len(self._cache)} sequence(s)")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _find_sequences(self, requested: Optional[List[str]]) -> List[str]:
        """Return sorted list of sequence names present in data_root."""
        found = []
        for p in sorted(self.data_root.glob('*_data.hdf5')):
            name = p.name.replace('_data.hdf5', '')
            flow_file = self.data_root / f'{name}_gt_flow_dist.npz'
            if flow_file.exists():
                found.append(name)

        if requested is not None:
            found = [s for s in found if s in requested]

        if len(found) == 0:
            raise FileNotFoundError(
                f"No valid MVSEC sequences found in {self.data_root}. "
                "Expected pairs of <name>_data.hdf5 and <name>_gt_flow_dist.npz"
            )
        return found

    def _build_sample_list(self) -> List[Dict]:
        samples = []
        for seq_name in self.sequence_names:
            cache = self._cache[seq_name]
            num_gt_frames = cache['gt_timestamps'].shape[0]

            # Determine valid index range
            if seq_name in self.filter:
                idx_start, idx_end = self.filter[seq_name]
                idx_end = min(idx_end, num_gt_frames - self.dt)
            else:
                idx_start = 0
                idx_end = num_gt_frames - self.dt

            is_outdoor = seq_name.startswith('outdoor')

            for i in range(idx_start, idx_end):
                samples.append({
                    'sequence': seq_name,
                    'index': i,
                    'is_outdoor': is_outdoor,
                })
        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        frame_idx = sample_info['index']
        cache = self._cache[sample_info['sequence']]

        # ---- Load ground-truth flow (from cached arrays) ----
        gt_ts = cache['gt_timestamps']
        x_flow_all = cache['x_flow_dist']
        y_flow_all = cache['y_flow_dist']

        # Accumulate flow over dt frames
        u_flow = np.zeros_like(x_flow_all[0])
        v_flow = np.zeros_like(y_flow_all[0])
        for d in range(self.dt):
            u_flow += x_flow_all[frame_idx + d]
            v_flow += y_flow_all[frame_idx + d]

        # Build validity mask: NaN/Inf in the GT marks invalid pixels
        valid = np.isfinite(u_flow) & np.isfinite(v_flow)
        u_flow = np.nan_to_num(u_flow, nan=0.0, posinf=0.0, neginf=0.0)
        v_flow = np.nan_to_num(v_flow, nan=0.0, posinf=0.0, neginf=0.0)

        # Mask outdoor car hood
        if self.mask_outdoor_hood and sample_info['is_outdoor']:
            valid[OUTDOOR_HOOD_ROW:, :] = False
            u_flow[OUTDOOR_HOOD_ROW:, :] = 0.0
            v_flow[OUTDOOR_HOOD_ROW:, :] = 0.0

        flow = torch.from_numpy(
            np.stack([u_flow, v_flow], axis=0)  # [2, H, W]
        ).float()
        valid_mask = torch.from_numpy(valid[np.newaxis, ...]).float()  # [1, H, W]

        # ---- Load events (coarse index → small HDF5 read → fine search) ----
        t_end = gt_ts[frame_idx + self.dt]
        time_window_s = self.num_bins * self.bin_interval_us / 1e6
        t_start = t_end - time_window_s

        # Step 1: use the image_raw_ts / image_raw_event_inds coarse index
        # to find the approximate event range, avoiding a full-array read.
        img_ts = cache['img_ts']
        img_ev_inds = cache['img_ev_inds']
        num_events = cache['num_events']

        # Find image frames bracketing our time window (with 1-frame margin)
        lo_img = max(0, np.searchsorted(img_ts, t_start, side='left') - 1)
        hi_img = min(len(img_ev_inds) - 1,
                     np.searchsorted(img_ts, t_end, side='right'))

        coarse_start = int(img_ev_inds[lo_img])
        coarse_end = int(img_ev_inds[hi_img]) if hi_img < len(img_ev_inds) \
                     else num_events

        # Step 2: read only the timestamps in that coarse range, then refine.
        with h5py.File(cache['data_path'], 'r') as h5:
            events_ds = h5['davis']['left']['events']
            chunk_ts = events_ds[coarse_start:coarse_end, 2]

            fine_start = np.searchsorted(chunk_ts, t_start, side='left')
            fine_end = np.searchsorted(chunk_ts, t_end, side='right')

            raw = events_ds[coarse_start + fine_start:
                            coarse_start + fine_end]  # [M, 4]

        # Columns: x, y, t (seconds), p (0 or 1)
        x = raw[:, 0].astype(np.float32)
        y = raw[:, 1].astype(np.float32)
        t = raw[:, 2].astype(np.float64)
        p = raw[:, 3].astype(np.float32)

        # Convert polarity 0/1 → -1/+1 (same convention as your dataset)
        p = np.where(p == 0, -1.0, 1.0).astype(np.float32)

        events = np.column_stack([x, y, t, p])  # [M, 4]

        input_tensor = self._events_to_voxel_grid(events)

        # ---- Center crop to crop_size if specified ----
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            _, _, H, W = input_tensor.shape
            y0 = (H - crop_h) // 2
            x0 = (W - crop_w) // 2
            input_tensor = input_tensor[:, :, y0:y0+crop_h, x0:x0+crop_w]
            flow = flow[:, y0:y0+crop_h, x0:x0+crop_w]
            valid_mask = valid_mask[:, y0:y0+crop_h, x0:x0+crop_w]

        # ---- Optional flow clipping ----
        if self.flow_clip_range is not None:
            flow = torch.clamp(flow, self.flow_clip_range[0], self.flow_clip_range[1])

        # ---- Augmentations (same API as OpticalFlowDataset) ----
        if self.augment_rotation:
            input_tensor, flow, valid_mask = self._apply_rotation_augmentation(
                input_tensor, flow, valid_mask
            )

        if self.flip_left_to_right_prob > 0:
            input_tensor, flow, valid_mask = self._maybe_flip_left_to_right(
                input_tensor, flow, valid_mask
            )

        if self.patch_mode:
            patch_data = self._extract_single_patch(input_tensor, flow, valid_mask)
            if self.return_full_frame:
                patch_data['full_input'] = input_tensor
                patch_data['full_flow'] = flow
                patch_data['full_valid_mask'] = valid_mask
            return patch_data

        return {
            'input': input_tensor,
            'flow': flow,
            'valid_mask': valid_mask,
            'metadata': {
                'sequence': sample_info['sequence'],
                'index': frame_idx,
                'dt': self.dt,
            }
        }

    # ------------------------------------------------------------------
    # Voxel grid construction (mirrors OpticalFlowDataset._events_to_voxel_grid)
    # ------------------------------------------------------------------
    def _events_to_voxel_grid(self, events: np.ndarray) -> torch.Tensor:
        """
        Convert [N, 4] events (x, y, t, p) into a voxel grid.

        Returns
            [num_bins, C, H, W]  where C=2 if use_polarity else C=1
        """
        height, width = self.data_size

        if len(events) == 0:
            c = 2 if self.use_polarity else 1
            return torch.zeros(self.num_bins, c, height, width)

        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        t = events[:, 2]
        p = events[:, 3]

        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min) * (self.num_bins - 1e-6)
        else:
            t_norm = np.zeros_like(t)

        if self.use_polarity:
            voxel = np.zeros((self.num_bins, 2, height, width), dtype=np.float32)
        else:
            voxel = np.zeros((self.num_bins, 1, height, width), dtype=np.float32)

        # Vectorised binning (replaces per-event Python loop)
        bin_idx = t_norm.astype(np.int32)
        mask = (bin_idx >= 0) & (bin_idx < self.num_bins) \
             & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        bin_idx = bin_idx[mask]
        xm = x[mask]
        ym = y[mask]
        pm = p[mask]

        if self.use_polarity:
            pol_idx = np.where(pm > 0, 0, 1).astype(np.int32)
            np.add.at(voxel, (bin_idx, pol_idx, ym, xm), 1.0)
        else:
            np.add.at(voxel, (bin_idx, 0, ym, xm), 1.0)

        return torch.from_numpy(voxel)

    # ------------------------------------------------------------------
    # Augmentations (identical to OpticalFlowDataset)
    # ------------------------------------------------------------------
    def _apply_rotation_augmentation(
        self,
        input_tensor: torch.Tensor,
        flow: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = np.random.randint(0, 4)
        if k == 0:
            return input_tensor, flow, valid_mask

        input_rotated = torch.rot90(input_tensor, k=k, dims=(-2, -1))
        valid_rotated = torch.rot90(valid_mask, k=k, dims=(-2, -1))
        flow_rotated = torch.rot90(flow, k=k, dims=(-2, -1))

        if k == 1:
            flow_rotated = torch.stack([-flow_rotated[1], flow_rotated[0]], dim=0)
        elif k == 2:
            flow_rotated = torch.stack([-flow_rotated[0], -flow_rotated[1]], dim=0)
        elif k == 3:
            flow_rotated = torch.stack([flow_rotated[1], -flow_rotated[0]], dim=0)

        return input_rotated, flow_rotated, valid_rotated

    def _maybe_flip_left_to_right(
        self,
        input_tensor: torch.Tensor,
        flow: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            denom = valid_mask.sum()
            if denom <= 0:
                return input_tensor, flow, valid_mask
            mean_u = (flow[0] * valid_mask[0]).sum() / denom
            if mean_u < self.flip_left_threshold:
                if torch.rand(1).item() < self.flip_left_to_right_prob:
                    input_tensor = torch.flip(input_tensor, dims=[-1])
                    flow = torch.flip(flow, dims=[-1])
                    flow[0] = -flow[0]
                    valid_mask = torch.flip(valid_mask, dims=[-1])
        return input_tensor, flow, valid_mask

    def _extract_single_patch(
        self,
        input_tensor: torch.Tensor,
        flow: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        _, _, H, W = input_tensor.shape
        half = self.patch_size // 2

        activity = input_tensor.sum(dim=(0, 1))
        flow_mag = torch.norm(flow, p=2, dim=0)

        candidates_y, candidates_x = torch.where(
            (activity >= self.activity_threshold) & (flow_mag > 0.1)
        )
        ok = (
            (candidates_y >= half) & (candidates_y < H - half)
            & (candidates_x >= half) & (candidates_x < W - half)
        )
        candidates_y, candidates_x = candidates_y[ok], candidates_x[ok]

        if len(candidates_y) == 0:
            candidates_y, candidates_x = torch.where(activity >= self.activity_threshold)
            ok = (
                (candidates_y >= half) & (candidates_y < H - half)
                & (candidates_x >= half) & (candidates_x < W - half)
            )
            candidates_y, candidates_x = candidates_y[ok], candidates_x[ok]

        if len(candidates_y) == 0:
            cy = torch.randint(half, H - half, (1,)).item()
            cx = torch.randint(half, W - half, (1,)).item()
        else:
            pick = torch.randint(0, len(candidates_y), (1,)).item()
            cy, cx = int(candidates_y[pick]), int(candidates_x[pick])

        return {
            'input': input_tensor[:, :, cy - half:cy + half, cx - half:cx + half],
            'flow': flow[:, cy - half:cy + half, cx - half:cx + half],
            'valid_mask': valid_mask[:, cy - half:cy + half, cx - half:cx + half],
            'metadata': {'patch_mode': True, 'center': (cy, cx)},
        }