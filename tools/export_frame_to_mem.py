"""
Export a single timestep from a dataset frame to a .mem file for Vivado simulation.

Format:
- Binary format, 32 bits per line
- LSB (bit 0) is on the right side
- 32 pixels per line
- Pixel (x=0) maps to bit 0, pixel (x=31) maps to bit 31
- For 320-wide image: 10 lines per image row

Example:
    python tools/export_frame_to_mem.py \
        --config snn/configs/event_snn_lite.yaml \
        --frame-idx 0 \
        --timestep 0 \
        --output frame.mem
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from snn.dataset.dataset import OpticalFlowDataset


def frame_to_mem_lines(frame: np.ndarray) -> list:
    """
    Convert a 2D frame to .mem format lines.
    
    Args:
        frame: 2D numpy array [H, W] with event counts
        
    Returns:
        List of 32-bit binary strings (MSB on left, LSB on right)
    """
    H, W = frame.shape
    pixels_per_line = 32
    
    # Binarize
    binary_frame = (frame > 0.5).astype(np.uint8)
    
    lines = []
    
    for y in range(H):
        row = binary_frame[y, :]
        
        # Process row in chunks of 32 pixels
        for chunk_start in range(0, W, pixels_per_line):
            chunk_end = min(chunk_start + pixels_per_line, W)
            chunk = row[chunk_start:chunk_end]
            
            # Build 32-bit value
            # Pixel at x=0 (within chunk) goes to bit 0 (LSB)
            value = 0
            for i, pixel in enumerate(chunk):
                if pixel:
                    value |= (1 << i)
            
            # Format as 32-bit binary string (MSB on left)
            binary_str = format(value, '032b')
            lines.append(binary_str)
    
    return lines


def main():
    parser = argparse.ArgumentParser(description="Export frame timestep to .mem file")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index in dataset")
    parser.add_argument("--timestep", type=int, default=0, help="Timestep index (0 to num_bins-1)")
    parser.add_argument("--output", default="frame.mem", help="Output .mem file path")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    dataset = OpticalFlowDataset(config)
    
    if args.frame_idx >= len(dataset):
        print(f"Error: frame_idx {args.frame_idx} >= dataset length {len(dataset)}")
        sys.exit(1)
    
    # Load frame
    sample = dataset[args.frame_idx]
    input_tensor = sample['input']  # [num_bins, 2, H, W]
    
    num_bins = input_tensor.shape[0]
    if args.timestep >= num_bins:
        print(f"Error: timestep {args.timestep} >= num_bins {num_bins}")
        sys.exit(1)
    
    # Extract single timestep and polarity channel
    frame = input_tensor[args.timestep, 0].numpy()  # [H, W]
    
    H, W = frame.shape
    print(f"Frame shape: {H}x{W}")
    print(f"Non-zero pixels: {np.count_nonzero(frame)}")
    
    # Convert to .mem format
    mem_lines = frame_to_mem_lines(frame)
    
    # Write output (no header)
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for line in mem_lines:
            f.write(line + '\n')
    
    print(f"Wrote {len(mem_lines)} lines to {output_path}")
    print(f"Expected lines: {H * (W // 32)} (H={H}, W={W}, 32 pixels/line)")
    
    # Generate index file with x,y coordinates of each event in row-major order
    binary_frame = (frame > 0.5).astype(np.uint8)
    index_path = output_path.with_suffix('.idx')
    event_count = 0
    with open(index_path, 'w') as f:
        for y in range(H):
            for x in range(W):
                if binary_frame[y, x]:
                    f.write(f"{x} {y}\n")
                    event_count += 1
    
    print(f"Wrote {event_count} events to {index_path}")
    
    # Display the frame
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw event counts
    im0 = axes[0].imshow(frame, cmap='hot', aspect='equal')
    axes[0].set_title(f'Event counts (timestep {args.timestep})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], label='count')
    
    # Binarized version (what goes into .mem)
    axes[1].imshow(binary_frame, cmap='gray', aspect='equal')
    axes[1].set_title(f'Binarized')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
