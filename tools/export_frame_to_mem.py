"""
Export a single timestep from a dataset frame to a .mem file for Vivado simulation.

Optionally loads a trained model and runs integer inference on the frame,
exporting per-layer debug data (accumulator, output feature maps, membrane
potentials) for FPGA verification.

Frame format:
- Binary format, 32 bits per line
- LSB (bit 0) is on the right side
- 32 pixels per line
- Pixel (x=0) maps to bit 0, pixel (x=31) maps to bit 31
- For 320-wide image: 10 lines per image row

Example (frame only):
    python tools/export_frame_to_mem.py \
        --config snn/configs/event_snn_lite_8bit.yaml \
        --frame-idx 0 --timestep 0 \
        --output frame.mem

Example (with integer inference debug):
    python tools/export_frame_to_mem.py \
        --config snn/configs/event_snn_lite_8bit.yaml \
        --checkpoint checkpoints/ptq_8bit/ptq_model.pth \
        --frame-idx 0 \
        --output output/debug_export/frame.mem
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from snn.dataset.dataset import OpticalFlowDataset
from utils import get_model, load_config


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


def write_feature_map_mem(path: Path, tensor: torch.Tensor, name: str,
                          as_float: bool = False):
    """Write a feature map tensor [N, C, H, W] or [C, H, W] to a .mem file.

    Values are written one per line (integers by default, floats if as_float=True).
    Order: channel 0 row-major, then channel 1, etc. (only sample 0 if batched).
    A comment line marks each channel.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # take sample 0: [C, H, W]
    C, H, W = tensor.shape
    lines = []
    for c in range(C):
        for y in range(H):
            for x in range(W):
                val = tensor[c, y, x].item()
                if as_float:
                    lines.append(f"ch={c}, x={x}, y={y}, v={val:.6f}")
                else:
                    lines.append(f"ch={c}, x={x}, y={y}, v={int(val)}")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def export_debug_data(debug_result: dict, output_dir: Path):
    """Export all debug intermediate tensors to .mem files in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_debug = debug_result['layer_debug']

    # Export input
    write_feature_map_mem(output_dir / 'input_int.mem',
                          debug_result['input_int'], 'input_int')

    # Export per-layer debug data
    for layer_name, data in layer_debug.items():
        layer_dir = output_dir / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        if 'sum' in data:
            write_feature_map_mem(layer_dir / 'sum.mem',
                                  data['sum'], f'{layer_name}/sum')

        if 'sum_prod' in data:
            write_feature_map_mem(layer_dir / 'sum_prod.mem',
                                  data['sum_prod'], f'{layer_name}/sum_prod')


        if 'fm_out' in data:
            write_feature_map_mem(layer_dir / 'fm_out.mem',
                                  data['fm_out'], f'{layer_name}/fm_out')

        if 'memb_pre' in data:
            write_feature_map_mem(layer_dir / 'memb_pre.mem',
                                  data['memb_pre'], f'{layer_name}/memb_pre')

        if 'memb_post' in data:
            write_feature_map_mem(layer_dir / 'memb_post.mem',
                                  data['memb_post'], f'{layer_name}/memb_post')

    # Export flow output
    write_feature_map_mem(output_dir / 'flow_int.mem',
                          debug_result['flow_int'], 'flow_int')
    write_feature_map_mem(output_dir / 'flow_float.mem',
                          debug_result['flow_float'], 'flow_float',
                          as_float=True)

    # Print summary
    print(f"\nDebug data exported to {output_dir}/")
    print(f"  input_int.mem")
    for layer_name, data in layer_debug.items():
        files = [k for k in ['sum', 'sum_prod', 'spike', 'memb_pre', 'memb_post'] if k in data]
        shapes = []
        for k in files:
            t = data[k]
            s = list(t.shape) if t.dim() <= 3 else list(t.shape[1:])
            shapes.append(f"{k}{s}")
        print(f"  {layer_name}/: {', '.join(shapes)}")
    print(f"  flow_int.mem, flow_float.mem")


def main():
    parser = argparse.ArgumentParser(description="Export frame to .mem file")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index in dataset")
    parser.add_argument("--output", default="frame.mem", help="Output .mem file path")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained model checkpoint for integer inference debug")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create dataset
    dataset = OpticalFlowDataset(config)

    if args.frame_idx >= len(dataset):
        print(f"Error: frame_idx {args.frame_idx} >= dataset length {len(dataset)}")
        sys.exit(1)

    # Load frame
    sample = dataset[args.frame_idx]
    input_tensor = sample['input']  # [num_bins, 2, H, W]
    num_bins = input_tensor.shape[0]

    # For legacy output: export the first timestep as the main .mem file
    frame = input_tensor[0, 0].numpy()  # [H, W]
    H, W = frame.shape
    print(f"Frame shape: {H}x{W}")
    print(f"Non-zero pixels: {np.count_nonzero(frame)}")

    # Convert to .mem format
    mem_lines = frame_to_mem_lines(frame)

    # Write output (no header)
    output_path = Path(f"{Path(args.output).parent}_{args.frame_idx}") / Path(args.output).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    # Save and display the frame as an image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from snn.utils.visualization import visualize_flow

    # --- Integer inference debug (optional) ---
    per_timestep_flows = []  # list of (flow_float, flow_int) per timestep
    if args.checkpoint:
        print(f"\n--- Running integer inference debug ---")
        from snn.models.integer_inference import IntegerInferenceModel

        device = 'cpu'

        # Build and load model
        model = get_model(config)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Build integer inference model
        int_model = IntegerInferenceModel(
            model, config,
            accum_bit_width=config.get('accum_bit_width', 32),
        )
        int_model.eval()

        # Run all timesteps, collecting per-timestep flow and debug data
        mems = None
        for t in range(num_bins):
            x_t = input_tensor[t].unsqueeze(0).float()  # [1, 2, H, W]
            result_t = int_model.forward_single_timestep(x_t, mems)
            mems = result_t['mems']
            # Move mems back to device for next timestep
            mems = {k: v.to(device) if v is not None else None
                    for k, v in mems.items()}

            per_timestep_flows.append((
                result_t['flow_float'][0],  # [2, H', W']
                result_t['flow_int'][0],    # [2, H', W']
            ))

            # Export debug data for every timestep into subfolders
            print(f"Input shape (timestep {t}): {list(x_t.shape)}")
            debug_dir = output_path.parent / str(t)
            export_debug_data(result_t, debug_dir)

        print(f"Ran inference for all {num_bins} timesteps")

    # --- Save frame image ---
    if per_timestep_flows:
        # Layout: events | binarized | t0 | t1 | ... | tN-1 | accumulated
        n_flow = len(per_timestep_flows)
        n_cols = 2 + n_flow + 1  # 2 input panels + per-timestep + accumulated
    else:
        n_cols = 2

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    im0 = axes[0].imshow(frame, cmap='hot', aspect='equal')
    axes[0].set_title(f'Event counts\n(frame {args.frame_idx})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], label='count', fraction=0.046)

    axes[1].imshow(binary_frame, cmap='gray', aspect='equal')
    axes[1].set_title('Binarized\n(input to .mem)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    if per_timestep_flows:
        flow_acc = None
        for t_idx, (flow_f, flow_i) in enumerate(per_timestep_flows):
            ax = axes[2 + t_idx]
            flow_vis = visualize_flow(flow_f)
            ax.imshow(flow_vis)
            ax.set_title(f'Flow t={t_idx}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            # Annotate min/max
            f_min, f_max = flow_f.min().item(), flow_f.max().item()
            i_min, i_max = flow_i.min().item(), flow_i.max().item()
            ax.text(0.02, 0.02,
                    f'f:[{f_min:.3f},{f_max:.3f}]\ni:[{i_min},{i_max}]',
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Accumulate flow
            if flow_acc is None:
                flow_acc = flow_f.clone()
            else:
                flow_acc = flow_acc + flow_f

        # Final accumulated flow panel
        ax_acc = axes[2 + len(per_timestep_flows)]
        flow_vis_acc = visualize_flow(flow_acc)
        ax_acc.imshow(flow_vis_acc)
        ax_acc.set_title(f'Accumulated\n(all {len(per_timestep_flows)} timesteps)')
        ax_acc.set_xlabel('x')
        ax_acc.set_ylabel('y')
        f_min, f_max = flow_acc.min().item(), flow_acc.max().item()
        ax_acc.text(0.02, 0.02,
                    f'f:[{f_min:.3f},{f_max:.3f}]',
                    transform=ax_acc.transAxes, fontsize=7,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    img_path = output_path.with_suffix('.png')
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"Saved frame image to {img_path}")


if __name__ == "__main__":
    main()
