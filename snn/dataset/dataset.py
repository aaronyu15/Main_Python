"""
Optical Flow Dataset Loader
Loads event data and optical flow from blink_sim outputs
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import h5py
import glob
import sys


class OpticalFlowDataset(Dataset):
    """
    Dataset for loading optical flow data from blink_sim output
    
    Expected structure:
    blink_sim/output/train/
        sequence_name_0/
            events_left/
                *.npy
            forward_flow/
                *.npy
            rgb_event_input/
                *.png
            rgb_reference/
                *.png
    """
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Args:
            config: Configuration dictionary containing all parameters. If provided,
                   individual arguments will override config values.
            data_root: Root directory containing the data (e.g., blink_sim/output)
            use_events: Use event data (vs RGB images)
            num_bins: Number of temporal bins for event representation
            data_size: Full image size (height, width)
            patch_mode: If True, extract patches from high-activity regions instead of full images
            patch_size: Size of patches to extract when patch_mode=True
            activity_threshold: Minimum event count to consider a region active
            flow_clip_range: Optional (min, max) to clip flow values
            min_event_count: Minimum total event count for a sample (retries if below threshold)
            return_full_frame: If True (and patch_mode=True), also return full frame data alongside patch
            augment_rotation: If True, apply random 90-degree rotations for data augmentation
        """
        # Use config as base, with individual args as overrides
        if config is None:
            config = {}
        
        self.data_root = config.get('data_root', '../blink_sim/output')
        self.use_events = config.get('use_events', True)
        self.num_bins = config.get('num_bins', 5)
        self.bin_interval_us = config.get('bin_interval_us', 10000)
        self.use_polarity = config.get('use_polarity', False)
        self.data_size = config.get('data_size', (320, 320))

        self.patch_mode = config.get('patch_mode', False)
        self.patch_size = config.get('patch_size', 64)
        self.activity_threshold = config.get('activity_threshold', 5)

        self.flow_clip_range = config.get('flow_clip_range', None)
        self.return_full_frame = config.get('return_full_frame', False)
        self.sparsify = config.get('sparsify', False)

        self.augment_rotation = config.get('augment_rotation', False)
        
        # Find all sequences
        self.sequences = self._find_sequences()
        
        # Build sample list
        self.samples = self._build_sample_list()

    
    def _find_sequences(self) -> List[Path]:
        split_dir = Path(self.data_root)
        
        sequences = []
        for seq_dir in sorted(split_dir.iterdir()):
            if seq_dir.is_dir() and (seq_dir / 'forward_flow').exists():
                sequences.append(seq_dir)
        
        return sequences
    
    def _build_sample_list(self) -> List[Dict]:
        samples = []
        
        for seq_dir in self.sequences:
            flow_dir = seq_dir / 'forward_flow'
            flow_files = sorted(flow_dir.glob('*.npy'))
            
            if self.use_events:
                event_dir = seq_dir / 'events_left'
                
                event_h5_file = event_dir / 'events.h5'
                if event_h5_file.exists():
                    num_frames = len(flow_files)
                    for flow_file in flow_files:
                        frame_idx = int(flow_file.stem)
                        samples.append({
                            'sequence': seq_dir.name,
                            'flow_path': flow_file,
                            'event_h5_path': event_h5_file,
                            'index': frame_idx,
                            'num_frames': num_frames
                        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample
        
        Returns:
            Dictionary containing:
                - 'input': Event representation or RGB image [C, H, W]
                - 'flow': Optical flow ground truth [2, H, W]
                - 'valid_mask': Valid flow mask [1, H, W]
                - 'metadata': Dictionary with sequence info
        """
        sample_info = self.samples[idx]
        
        # Load optical flow
        flow_data = np.load(sample_info['flow_path'])  # [H, W, 2] or [H, W, 3]
        
        # Format: [u, v, valid] - extract flow and validity
        flow = torch.from_numpy(flow_data[:, :, :2]).permute(2, 0, 1).float()  # [2, H, W]
        valid_mask = torch.from_numpy(flow_data[:, :, 2:3]).permute(2, 0, 1).float()  # [1, H, W]
        
        frame_idx = sample_info['index']

        try:
            with h5py.File(sample_info['event_h5_path'], 'r') as f:
                all_t = f['events/t'][:]

                total_duration_us = all_t[-1] - all_t[0]
                t_start_us = all_t[0]

                num_flow_frames = sample_info.get('num_frames', 30)
                frame_interval_us = total_duration_us / max(1, num_flow_frames)

                time_window_us = self.num_bins * self.bin_interval_us

                t1 = t_start_us + (frame_idx + 1) * frame_interval_us
                t0 = t1 - time_window_us

                start_idx = np.searchsorted(all_t, t0, side='left')
                end_idx = np.searchsorted(all_t, t1, side='left')

                x = np.array(f['events/x'][start_idx:end_idx])
                y = np.array(f['events/y'][start_idx:end_idx])
                t = np.array(f['events/t'][start_idx:end_idx])
                p = np.array(f['events/p'][start_idx:end_idx])

                p = np.where(p == 0, -1, 1)

                events = np.column_stack([x, y, t, p]).astype(np.float32)
        except Exception as e:
            print(f"Warning: Failed to load events for frame {frame_idx}: {e}")
            sys.exit(1)
            

        input_tensor = self._events_to_voxel_grid(events)  # [num_bins, H, W]
        
        # Note: valid_mask was already created when loading flow
        # If not created (shouldn't happen), create default
        if 'valid_mask' not in locals():
            valid_mask = torch.ones(1, flow.shape[1], flow.shape[2])

        if self.sparsify:
            # Sparsify input patch by zeroing out low-activity bins
            activity_patch = input_tensor.sum(dim=(0, 1))
            low_activity_mask = (activity_patch < 1)

            input_tensor[:, :, low_activity_mask] = 0.0
            valid_mask[:, low_activity_mask] = 0.0
        # Apply flow clipping if specified
        if self.flow_clip_range is not None:
            flow = torch.clamp(flow, self.flow_clip_range[0], self.flow_clip_range[1])
    
        # Apply random rotation augmentation if enabled
        if self.augment_rotation:
            input_tensor, flow, valid_mask = self._apply_rotation_augmentation(
                input_tensor, flow, valid_mask
            )
    
        # Extract single random patch from high-activity regions if in patch mode
        if self.patch_mode:
            patch_data = self._extract_single_patch(
                input_tensor, flow, valid_mask
            )

            # Optionally include full frame data
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
                'index': sample_info['index']
            }
        }
    
    def _events_to_voxel_grid(self, events: np.ndarray) -> torch.Tensor:
        """
        Convert event array to voxel grid representation
        
        Events format: [N, 4] where columns are [x, y, t, p]
        - x, y: pixel coordinates
        - t: timestamp
        - p: polarity (+1 or -1)
        
        Returns:
            Voxel grid [num_bins, 2, H, W] for EventSNNFlowNetLite (polarity-separated)
            OR [num_bins, H, W] for other models (mixed polarity)
        """
        if len(events) == 0:
            # Return zeros if no events - use polarity-separated format
            if self.use_polarity:
                return torch.zeros(self.num_bins, 2, 320, 320)  # Default size
            else:
                return torch.zeros(self.num_bins, 1, 320, 320)  # Default size      
        
        # Parse events
        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        t = events[:, 2]
        p = events[:, 3]
        
        # Get image dimensions
        height = self.data_size[0]
        width = self.data_size[1]
        
        # Normalize timestamps to [0, num_bins)
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min) * (self.num_bins - 1e-6)
        else:
            t_norm = np.zeros_like(t)
        
        # Create voxel grid with separate polarity channels [num_bins, 2, H, W]
        # Channel 0 = positive events, Channel 1 = negative events
        if self.use_polarity:
            voxel_grid = np.zeros((self.num_bins, 2, height, width), dtype=np.float32)
        else:
            voxel_grid = np.zeros((self.num_bins, 1, height, width), dtype=np.float32)
        
        # Distribute events into temporal bins with polarity separation
        for i in range(len(events)):
            bin_idx = int(t_norm[i])
            if 0 <= bin_idx < self.num_bins:
                if 0 <= x[i] < width and 0 <= y[i] < height:
                    if self.use_polarity:
                        pol_idx = 0 if p[i] > 0 else 1  # Channel 0 = positive, 1 = negative
                    else:
                        pol_idx = 0 
                    voxel_grid[bin_idx, pol_idx, y[i], x[i]] += 1.0
        
        return torch.from_numpy(voxel_grid)
    
    def _apply_rotation_augmentation(
        self,
        input_tensor: torch.Tensor,
        flow: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random 90-degree rotation augmentation to input, flow, and valid mask
        
        Rotations are k*90 degrees (k=0,1,2,3) which preserves data quality and is efficient.
        
        Args:
            input_tensor: Event voxel grid [num_bins, C, H, W] or RGB [3, H, W]
            flow: Optical flow [2, H, W] where [0] is u (horizontal), [1] is v (vertical)
            valid_mask: Valid flow mask [1, H, W]
        
        Returns:
            Rotated (input_tensor, flow, valid_mask)
        """
        # Randomly choose rotation: 0, 90, 180, or 270 degrees
        k = np.random.randint(0, 4)
        
        if k == 0:
            # No rotation
            return input_tensor, flow, valid_mask
        
        # Rotate spatial dimensions (k*90 degrees counter-clockwise)
        # torch.rot90 rotates in the plane specified by dims, k times counter-clockwise
        input_rotated = torch.rot90(input_tensor, k=k, dims=(-2, -1))
        valid_rotated = torch.rot90(valid_mask, k=k, dims=(-2, -1))
        
        # Rotate flow field - both spatial dimensions AND flow vectors
        flow_rotated = torch.rot90(flow, k=k, dims=(-2, -1))
        
        # Adjust flow vectors based on rotation
        # Flow is [u, v] where u=horizontal (x-direction), v=vertical (y-direction)
        # For 90° CCW:  (u, v) -> (-v, u)   [right becomes up, up becomes left]
        # For 180°:     (u, v) -> (-u, -v)  [right becomes left, up becomes down]
        # For 270° CCW: (u, v) -> (v, -u)   [right becomes down, up becomes right]
        
        if k == 1:  # 90° CCW
            flow_rotated = torch.stack([-flow_rotated[1], flow_rotated[0]], dim=0)
        elif k == 2:  # 180°
            flow_rotated = torch.stack([-flow_rotated[0], -flow_rotated[1]], dim=0)
        elif k == 3:  # 270° CCW (or 90° CW)
            flow_rotated = torch.stack([flow_rotated[1], -flow_rotated[0]], dim=0)
        
        return input_rotated, flow_rotated, valid_rotated
    
    def _extract_single_patch(
        self,
        input_tensor: torch.Tensor,
        flow: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract a single random patch from regions with high activity AND high flow
        
        Args:
            input_tensor: [num_bins, 2, H, W] event voxel grid
            flow: [2, H, W] optical flow
            valid_mask: [1, H, W] validity mask
        
        Returns:
            Dictionary with single patch
        """
        _, _, H, W = input_tensor.shape

        half = self.patch_size // 2
        
        # Compute activity map by summing over time and polarity
        activity = input_tensor.sum(dim=(0, 1))  # [H, W]
        
        # Compute flow magnitude map
        flow_mag = torch.norm(flow, p=2, dim=0)  # [H, W]
        
        # Find candidate locations with BOTH high activity AND high flow
        # This ensures we extract patches from regions with meaningful motion
        candidates_y, candidates_x = torch.where(
            (activity >= self.activity_threshold) & (flow_mag > 0.1)
        )
        
        # Filter to keep only positions not too close to borders
        valid_positions = (
            (candidates_y >= half) & (candidates_y < H - half) &
            (candidates_x >= half) & (candidates_x < W - half)
        )
        candidates_y = candidates_y[valid_positions]
        candidates_x = candidates_x[valid_positions]
        
        # If not enough candidates with high flow, try just high activity
        if len(candidates_y) == 0:
            candidates_y, candidates_x = torch.where(activity >= self.activity_threshold)
            valid_positions = (
                (candidates_y >= half) & (candidates_y < H - half) &
                (candidates_x >= half) & (candidates_x < W - half)
            )
            candidates_y = candidates_y[valid_positions]
            candidates_x = candidates_x[valid_positions]
        
        # If still no candidates, use random position
        if len(candidates_y) == 0:
            # Sample random valid position
            y = torch.randint(half, H - half, (1,)).item()
            x = torch.randint(half, W - half, (1,)).item()
        else:
            # Sample one from high-activity+flow regions
            idx = torch.randint(0, len(candidates_y), (1,)).item()
            y = int(candidates_y[idx])
            x = int(candidates_x[idx])
        
        # Extract patch from input
        input_patch = input_tensor[:, :, y-half:y+half, x-half:x+half]  # [T, 2, P, P]
        
        # Extract flow field for entire patch
        flow_patch = flow[:, y-half:y+half, x-half:x+half]  # [2, P, P]
        
        # Valid mask for entire patch
        valid_patch = valid_mask[:, y-half:y+half, x-half:x+half]  # [1, P, P]

        if self.sparsify:
            # Sparsify input patch by zeroing out low-activity bins
            activity_patch = input_patch.sum(dim=(0, 1))  # [P, P]
            low_activity_mask = (activity_patch < 1)

            input_patch[:, :, low_activity_mask] = 0.0
            valid_patch[:, low_activity_mask] = 0.0

        return {
            'input': input_patch,      # [T, 2, P, P]
            'flow': flow_patch,         # [2, P, P]
            'valid_mask': valid_patch,  # [1, P, P]
            'metadata': {'patch_mode': True, 'center': (y, x)}
        }

    