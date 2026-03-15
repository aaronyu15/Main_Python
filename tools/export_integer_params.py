"""
Export integer model parameters to .mem files for FPGA synthesis.

Exports weights, M_0, shift, and thresholds from a calibrated quantized
checkpoint into separate .mem files with one value per line.

All SNN layers (e1–d1) are concatenated into single files:
  weights.mem, m0.mem, shift.mem, threshold.mem
Flow head is exported separately:
  flow_head_weights.mem, flow_head_m0.mem, flow_head_shift.mem

Layer order: e1, e2, e3, e4, d4, d3, d2, d1.
Weight ordering within each layer: cin outer, cout inner, then ky, kx.
  i.e. all output filters for input ch 0, then all for input ch 1, etc.

Example:
    python tools/export_integer_params.py \
        --checkpoint checkpoints/ptq_8bit/ptq_model.pth \
        --config snn/configs/event_snn_lite_8bit.yaml \
        --output-dir output/mem_export
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from snn.models import EventSNNFlowNetLite
from snn.models.quant_utils import export_quantized_params
from utils import get_model, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export integer parameters to .mem files")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to quantized model checkpoint")
    parser.add_argument("--config", required=True,
                        help="Path to quantization config YAML")
    parser.add_argument("--output-dir", default="./output/mem_export",
                        help="Directory to write .mem files")
    parser.add_argument("--format", choices=["dec", "hex"], default="dec",
                        help="Number format: dec (signed decimal) or hex (two's complement)")
    parser.add_argument("--hex-width", type=int, default=None,
                        help="Hex digit width (auto-sized from bit-width if omitted)")
    return parser.parse_args()


def format_value(val: int, fmt: str, hex_width: int) -> str:
    """Format a single integer value as decimal or two's complement hex."""
    if fmt == "dec":
        return str(val)
    if val < 0:
        val = val + (1 << (hex_width * 4))
    return f"{val:0{hex_width}X}"


def pack_kernel_hex(kernel_vals, bits_per_weight=8):
    """Pack 9 kernel weights into one hex line, little-endian.

    kernel_vals: list of 9 signed integers (kH*kW, e.g. 3x3)
    Little-endian: kernel_vals[0] is the least significant byte.
    Each weight is two's complement with bits_per_weight bits.
    """
    mask = (1 << bits_per_weight) - 1
    packed = 0
    for i, w in enumerate(kernel_vals):
        # Two's complement for negative values
        if w < 0:
            w = w + (1 << bits_per_weight)
        packed |= (w & mask) << (i * bits_per_weight)
    n_hex = (len(kernel_vals) * bits_per_weight + 3) // 4
    return f"{packed:0{n_hex}X}"


def write_mem_file(path: Path, values, fmt: str, hex_width: int,
                   header: str = ""):
    """Write a list of integer values to a .mem file, one per line."""
    with open(path, "w") as f:
        if header:
            for line in header.strip().split("\n"):
                f.write(f"// {line}\n")
        for v in values:
            f.write(format_value(int(v), fmt, hex_width) + "\n")


def write_packed_weight_hex(path: Path, weight_tensors, bits_per_weight=8):
    """Write packed hex weight file. Each line is one 3x3 kernel (9 weights).

    weight_tensors: list of (layer_name, tensor[Cout, Cin/g, kH, kW])
    Order: for each layer, for each cin, for each cout: one line of 9 packed weights.
    Little-endian: w[0,0] at the least significant position.
    """
    with open(path, "w") as f:
        for layer_name, w in weight_tensors:
            cout, cin_g, kh, kw = w.shape
            for ci in range(cin_g):
                for co in range(cout):
                    kernel = w[co, ci].flatten().tolist()  # [kH*kW]
                    f.write(pack_kernel_hex(kernel, bits_per_weight) + "\n")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = "cpu"

    # Build and load model
    model = get_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Export quantized parameters
    multiplier_bits = config.get("multiplier_bits", 16)
    exported = export_quantized_params(model, multiplier_bits=multiplier_bits)
    if not exported:
        print("ERROR: No quantized layers found. Is this a calibrated model?")
        sys.exit(1)

    weight_bit_width = config.get("weight_bit_width", 8)
    act_bit_width = config.get("act_bit_width", 8)
    hex_width_w = args.hex_width or ((weight_bit_width + 3) // 4)
    hex_width_m0 = args.hex_width or ((multiplier_bits + 3) // 4)
    hex_width_shift = args.hex_width or 2
    hex_width_thresh = args.hex_width or ((act_bit_width + 3) // 4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SNN layer order (excludes flow_head)
    snn_layer_order = ["e1", "e2", "e3", "e4", "d4", "d3", "d2"]

    # Map exported conv names to layer names
    layer_params = {}
    for conv_name, params in exported.items():
        layer_name = conv_name.rsplit(".conv", 1)[0] if ".conv" in conv_name else conv_name
        layer_params[layer_name] = params

    fmt = args.format

    # ---- Concatenated SNN layer files ----
    all_weights = []
    all_m0 = []
    all_shift = []
    all_thresh = []
    weight_header_lines = []
    weight_tensors = []  # [(layer_name, tensor)] for hex packing
    total_weights = 0

    for layer_name in snn_layer_order:
        if layer_name not in layer_params:
            print(f"  [WARN] {layer_name} not found in exported params, skipping")
            continue

        params = layer_params[layer_name]

        # Weights
        w = params["int_weight"]
        cout, cin_g, kh, kw = w.shape
        w_flat = w.flatten().tolist()
        weight_header_lines.append(
            f"{layer_name}: [{cout}, {cin_g}, {kh}, {kw}] = {len(w_flat)} values "
            f"(offset {total_weights})")
        total_weights += len(w_flat)
        all_weights.extend(w_flat)
        weight_tensors.append((layer_name, w))

        # M_0
        all_m0.extend(params["M_0"].flatten().tolist())

        # Shift
        all_shift.extend(params["shift"].flatten().tolist())

        # Threshold
        if "threshold_int" in params:
            all_thresh.extend(params["threshold_int"].flatten().tolist())

        print(f"  {layer_name}: weights={len(w_flat)}, "
              f"m0={cout}, shift={cout}"
              + (f", threshold={len(params['threshold_int'].flatten().tolist())}"
                 if "threshold_int" in params else ""))

    # Write concatenated SNN files
    w_header = "SNN weights (all layers concatenated)\n" + "\n".join(weight_header_lines)
    w_header += f"\nTotal: {total_weights} values"
    write_mem_file(out_dir / "weights.mem", all_weights, fmt, hex_width_w,
                   header=w_header)

    m0_header = f"SNN M_0 values (all layers concatenated)\nTotal: {len(all_m0)} values"
    write_mem_file(out_dir / "m0.mem", all_m0, fmt, hex_width_m0,
                   header=m0_header)

    shift_header = f"SNN shift values (all layers concatenated)\nTotal: {len(all_shift)} values"
    write_mem_file(out_dir / "shift.mem", all_shift, fmt, hex_width_shift,
                   header=shift_header)

    if all_thresh:
        thresh_header = f"SNN threshold values (all layers concatenated)\nTotal: {len(all_thresh)} values"
        write_mem_file(out_dir / "threshold.mem", all_thresh, fmt,
                       hex_width_thresh, header=thresh_header)

    # ---- Hex files (packed weights, scalar hex for m0/shift/threshold) ----
    hex_dir = out_dir / "hex"
    hex_dir.mkdir(parents=True, exist_ok=True)

    # Packed hex weights: 9 weights per line, little-endian
    total_kernels = sum(w.shape[0] * w.shape[1] for _, w in weight_tensors)
    w_hex_header = (
        "SNN weights — packed hex, 9 weights per line (one 3x3 kernel)\n"
        "Little-endian: w[0,0] at LSB, w[2,2] at MSB\n"
        "Each weight: signed 8-bit two's complement (2 hex digits)\n"
        + "\n".join(weight_header_lines)
        + f"\nTotal: {total_kernels} kernels ({total_weights} weights)")
    write_packed_weight_hex(hex_dir / "weights.mem", weight_tensors,
                            bits_per_weight=weight_bit_width)

    # Scalar hex files for m0, shift, threshold
    write_mem_file(hex_dir / "m0.mem", all_m0, "hex", hex_width_m0)
    write_mem_file(hex_dir / "shift.mem", all_shift, "hex", hex_width_shift)
    if all_thresh:
        write_mem_file(hex_dir / "threshold.mem", all_thresh, "hex",
                       hex_width_thresh)

    # ---- Flow head (separate files) ----
    if "flow_head" in layer_params:
        params = layer_params["flow_head"]
        w = params["int_weight"]
        cout, cin_g, kh, kw = w.shape
        w_flat = w.permute(1, 0, 2, 3).flatten().tolist()

        fh_w_header = (f"flow_head weights\n"
                       f"Shape: [{cout}, {cin_g}, {kh}, {kw}]\n"
                       f"Total: {len(w_flat)} values")
        write_mem_file(out_dir / "flow_head_weights.mem", w_flat, fmt,
                       hex_width_w, header=fh_w_header)

        fh_m0 = params["M_0"].flatten().tolist()
        write_mem_file(out_dir / "flow_head_m0.mem", fh_m0, fmt,
                       hex_width_m0,
                       header=f"flow_head M_0\nShape: [{len(fh_m0)}]")

        fh_shift = params["shift"].flatten().tolist()
        write_mem_file(out_dir / "flow_head_shift.mem", fh_shift, fmt,
                       hex_width_shift,
                       header=f"flow_head shift\nShape: [{len(fh_shift)}]")

        print(f"  flow_head: weights={len(w_flat)}, "
              f"m0={len(fh_m0)}, shift={len(fh_shift)}")

        # Flow head hex files
        fh_w = params["int_weight"]
        fh_hex_header = (
            f"flow_head weights — packed hex, 9 weights per line\n"
            f"Shape: [{cout}, {cin_g}, {kh}, {kw}]")
        write_packed_weight_hex(hex_dir / "flow_head_weights.mem",
                                [("", fh_w)],
                                bits_per_weight=weight_bit_width)
        write_mem_file(hex_dir / "flow_head_m0.mem", fh_m0, "hex",
                       hex_width_m0)
        write_mem_file(hex_dir / "flow_head_shift.mem", fh_shift, "hex",
                       hex_width_shift)

    # Summary
    print(f"\nExported to {out_dir}/")
    print(f"  Decimal: weights.mem ({total_weights}), m0.mem ({len(all_m0)}), "
          f"shift.mem ({len(all_shift)}), threshold.mem ({len(all_thresh)})")
    print(f"  Hex:     hex/weights.mem ({total_kernels} lines), hex/m0.mem, "
          f"hex/shift.mem, hex/threshold.mem")
    print(f"  Flow head: flow_head_*.mem (decimal + hex)")


if __name__ == "__main__":
    main()
