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
import glob


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
        data_root: str,
        split: str = 'train',
        transform=None,
        use_events: bool = True,
        num_bins: int = 5,
        data_size: Tuple[int, int] = (320, 320),
        crop_size: Tuple[int, int] = (320, 320),
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_root: Root directory containing the data (e.g., blink_sim/output)
            split: 'train' or 'val'
            transform: Optional transform to apply
            use_events: Use event data (vs RGB images)
            num_bins: Number of temporal bins for event representation
            crop_size: Size to crop images/events to
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.use_events = use_events
        self.num_bins = num_bins
        self.crop_size = crop_size
        self.data_size = data_size
        
        # Find all sequences
        self.sequences = self._find_sequences()
        
        # Build sample list
        self.samples = self._build_sample_list()
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from {len(self.sequences)} sequences")
    
    def _find_sequences(self) -> List[Path]:
        """Find all sequence directories"""
        split_dir = self.data_root
        
        sequences = []
        for seq_dir in sorted(split_dir.iterdir()):
            if seq_dir.is_dir() and (seq_dir / 'forward_flow').exists():
                sequences.append(seq_dir)
        
        return sequences
    
    def _build_sample_list(self) -> List[Dict]:
        """Build list of all samples across sequences"""
        samples = []
        
        for seq_dir in self.sequences:
            # Get flow files
            flow_dir = seq_dir / 'forward_flow'
            flow_files = sorted(flow_dir.glob('*.npy'))
            
            # Get event or image files
            if self.use_events:
                event_dir = seq_dir / 'events_left'
                
                # Check if events are in HDF5 format or individual npy files
                event_h5_file = event_dir / 'events.h5'
                if event_h5_file.exists():
                    # Events stored in HDF5 - each flow file is a valid sample
                    num_frames = len(flow_files)
                    for flow_file in flow_files:
                        frame_idx = int(flow_file.stem)
                        samples.append({
                            'sequence': seq_dir.name,
                            'flow_path': flow_file,
                            'event_h5_path': event_h5_file,
                            'index': frame_idx,
                            'num_frames': num_frames  # Total frames in sequence for time interpolation
                        })
                else:
                    # Events stored as individual npy files
                    event_files = sorted(event_dir.glob('*.npy'))
                    
                    # Match flow and events
                    for flow_file in flow_files:
                        frame_idx = int(flow_file.stem)
                        event_file = event_dir / f"{frame_idx:06d}.npy"
                        
                        if event_file.exists():
                            samples.append({
                                'sequence': seq_dir.name,
                                'flow_path': flow_file,
                                'event_path': event_file,
                                'index': frame_idx
                            })
            else:
                # Use RGB images
                rgb_dir = seq_dir / 'rgb_event_input'
                rgb_files = sorted(rgb_dir.glob('*.png'))
                
                for flow_file in flow_files:
                    frame_idx = int(flow_file.stem)
                    rgb_file = rgb_dir / f"{frame_idx:06d}.png"
                    
                    if rgb_file.exists():
                        samples.append({
                            'sequence': seq_dir.name,
                            'flow_path': flow_file,
                            'rgb_path': rgb_file,
                            'index': frame_idx
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
        
        # Handle different flow formats
        if flow_data.shape[2] == 3:
            # Format: [u, v, valid] - extract flow and validity
            flow = torch.from_numpy(flow_data[:, :, :2]).permute(2, 0, 1).float()  # [2, H, W]
            valid_mask = torch.from_numpy(flow_data[:, :, 2:3]).permute(2, 0, 1).float()  # [1, H, W]
        elif flow_data.shape[2] == 2:
            # Format: [u, v] - assume all valid
            flow = torch.from_numpy(flow_data).permute(2, 0, 1).float()  # [2, H, W]
            valid_mask = torch.ones(1, flow_data.shape[0], flow_data.shape[1])
        else:
            raise ValueError(f"Unexpected flow shape: {flow_data.shape}")
        
        # Load input (events or RGB)
        if self.use_events:
            # Check if events are in HDF5 or individual npy files
            if 'event_h5_path' in sample_info:
                # Load from HDF5
                import h5py
                frame_idx = sample_info['index']
                
                try:
                    with h5py.File(sample_info['event_h5_path'], 'r') as f:
                        # Load all events' timestamps to determine frame boundaries
                        all_t = f['events/t'][:]
                        
                        # Compute time window for this frame
                        # Assume flow FPS (typically 30-60 fps from total duration)
                        if len(all_t) > 0:
                            total_duration_us = all_t[-1] - all_t[0]
                            t_start_us = all_t[0]
                            
                            # Get number of flow frames from sequence info
                            # Flow frames typically at ~30 fps, so compute interval
                            num_flow_frames = sample_info.get('num_frames', 60)
                            frame_interval_us = total_duration_us / max(1, num_flow_frames - 1)
                            
                            # Base time per bin (using 5 bins as reference = 1 frame interval)
                            # Each bin represents frame_interval_us / 5 microseconds
                            # So total window = (num_bins / 5) * frame_interval_us
                            # This means: 5 bins = 1 frame, 8 bins = 1.6 frames, 10 bins = 2 frames
                            time_window_us = (self.num_bins / 5.0) * frame_interval_us
                            
                            # Time window ends at the current frame time
                            t1 = t_start_us + (frame_idx + 1) * frame_interval_us
                            t0 = t1 - time_window_us
                            
                            # Find events in this time window using binary search
                            start_idx = np.searchsorted(all_t, t0, side='left')
                            end_idx = np.searchsorted(all_t, t1, side='left')
                            
                            # Load events in this time range
                            x = np.array(f['events/x'][start_idx:end_idx])
                            y = np.array(f['events/y'][start_idx:end_idx])
                            t = np.array(f['events/t'][start_idx:end_idx])
                            p = np.array(f['events/p'][start_idx:end_idx])
                            
                            # Convert polarity to -1/+1 if it's 0/1
                            p = np.where(p == 0, -1, 1)
                            
                            # Combine into single array [N, 4] with columns [x, y, t, p]
                            events = np.column_stack([x, y, t, p]).astype(np.float32)
                        else:
                            
                            events = np.column_stack([
                                x[start:end],
                                y[start:end],
                                t[start:end],
                                p[start:end]
                            ]).astype(np.float32)
                except Exception as e:
                    # If loading fails, create empty events
                    print(f"Warning: Failed to load events for frame {frame_idx}: {e}")
                    events = np.zeros((0, 4), dtype=np.float32)
                
                input_tensor = self._events_to_voxel_grid(events)  # [num_bins, H, W]
            else:
                # Load from individual npy file
                events = np.load(sample_info['event_path'])  # Event array
                input_tensor = self._events_to_voxel_grid(events)  # [num_bins, H, W]
        else:
            # Load RGB image
            from PIL import Image
            rgb = Image.open(sample_info['rgb_path']).convert('RGB')
            rgb = np.array(rgb).astype(np.float32) / 255.0
            input_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()  # [3, H, W]
        
        # Note: valid_mask was already created when loading flow
        # If not created (shouldn't happen), create default
        if 'valid_mask' not in locals():
            valid_mask = torch.ones(1, flow.shape[1], flow.shape[2])
        
        # Apply transforms (cropping, augmentation, etc.)
        if self.transform is not None:
            input_tensor, flow, valid_mask = self.transform(input_tensor, flow, valid_mask)
        
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
            return torch.zeros(self.num_bins, 2, 320, 320)  # Default size
        
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
        voxel_grid = np.zeros((self.num_bins, 2, height, width), dtype=np.float32)
        
        # Distribute events into temporal bins with polarity separation
        for i in range(len(events)):
            bin_idx = int(t_norm[i])
            if 0 <= bin_idx < self.num_bins:
                if 0 <= x[i] < width and 0 <= y[i] < height:
                    pol_idx = 0 if p[i] > 0 else 1  # Channel 0 = positive, 1 = negative
                    voxel_grid[bin_idx, pol_idx, y[i], x[i]] += 1.0
        
        return torch.from_numpy(voxel_grid)
    
    def _center_crop(self, input_tensor, flow, valid_mask):
        """Apply center crop/resize to tensors to match target crop_size"""
        import torch.nn.functional as F
        
        # Handle both 3D [C, H, W] and 4D [T, 2, H, W] input formats
        if input_tensor.dim() == 4:
            # 4D format [T, 2, H, W] for EventSNNFlowNetLite
            _, _, h, w = input_tensor.shape
        else:
            # 3D format [C, H, W] for other models
            _, h, w = input_tensor.shape
            
        crop_h, crop_w = self.crop_size
        
        # Resize to target size using bilinear interpolation
        if h != crop_h or w != crop_w:
            # Resize input - handle different dimensionalities
            if input_tensor.dim() == 4:
                # [T, 2, H, W] -> reshape to [T*2, H, W] for interpolation
                T, C_pol, _, _ = input_tensor.shape
                input_flat = input_tensor.view(T * C_pol, h, w)
                input_resized = F.interpolate(
                    input_flat.unsqueeze(0), 
                    size=(crop_h, crop_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                input_tensor = input_resized.view(T, C_pol, crop_h, crop_w)
            else:
                # [C, H, W] format
                input_tensor = F.interpolate(
                    input_tensor.unsqueeze(0), 
                    size=(crop_h, crop_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Resize flow and scale appropriately
            scale_h = crop_h / h
            scale_w = crop_w / w
            flow_resized = F.interpolate(
                flow.unsqueeze(0), 
                size=(crop_h, crop_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Scale flow values
            flow_resized[0] *= scale_w  # Scale x component
            flow_resized[1] *= scale_h  # Scale y component
            flow = flow_resized
            
            # Resize mask
            valid_mask = F.interpolate(
                valid_mask.unsqueeze(0), 
                size=(crop_h, crop_w), 
                mode='nearest'
            ).squeeze(0)
            
            h, w = crop_h, crop_w
        
        # Center crop
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        if input_tensor.dim() == 4:
            input_tensor = input_tensor[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        else:
            input_tensor = input_tensor[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        flow = flow[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        valid_mask = valid_mask[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        return input_tensor, flow, valid_mask


class EventDataset(Dataset):
    """
    Simplified event-based dataset for event-driven processing
    Directly loads event files without heavy preprocessing
    """
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        sequence_length: int = 10,
        dt: float = 0.01  # Time delta between frames
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.sequence_length = sequence_length
        self.dt = dt
        
        # Find sequences
        self.sequences = self._find_sequences()
        print(f"Found {len(self.sequences)} sequences for {split}")
    
    def _find_sequences(self) -> List[Path]:
        """Find all sequence directories"""
        split_dir = self.data_root / self.split
        
        sequences = []
        for seq_dir in sorted(split_dir.iterdir()):
            if seq_dir.is_dir() and (seq_dir / 'events_left').exists():
                sequences.append(seq_dir)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """Load event sequence"""
        seq_dir = self.sequences[idx]
        
        # Load events
        event_files = sorted((seq_dir / 'events_left').glob('*.npy'))
        events = [np.load(f) for f in event_files[:self.sequence_length]]
        
        # Load corresponding flows
        flow_files = sorted((seq_dir / 'forward_flow').glob('*.npy'))
        flows = [np.load(f) for f in flow_files[:self.sequence_length]]
        
        return {
            'events': events,
            'flows': flows,
            'sequence_name': seq_dir.name
        }
