"""
Utility to emit hardware instruction memory files (e.g., instruct.mem) from an
EventSNNFlowNetLite definition.

The generator walks the model layers in order, derives the convolution
parameters (channels, groups), estimates the output feature map
sizes, and allocates weight/membrane address ranges. All heuristics are exposed
via CLI flags so you can tweak packing/alignment without editing code.

Example:
    python tools/generate_instruct_mem.py \
        --config snn/configs/event_snn_lite.yaml \
        --output ../Main/rtl/src/mem/instruct.mem \
        --input-size 320 320 \
        --weight-base 0x0 --weight-align 16 --mem-pack 16 \
        --reuse-last-mem-for-head

Assumptions (matching the updated forward pass: stride-1 convs with explicit
pool/upsample):
- Weight count per layer = out_c * (in_c / groups) * k * k.
- One weight memory word stores 9 weights -> num_weight_addr = ceil(weight_count / 9).
- Membrane packing stores `mem_pack` activations per memory word ->
    num_mem_addr = ceil(out_h * out_w * out_c / mem_pack), where out_h/out_w are
    the conv outputs (pre-pooling/upsampling); pooling/upsampling only affects
    the next layer's input size.
- Weight addresses are aligned to `weight_align` (default 16) starting at
    `weight_base` (default 0x0). Membrane addresses are contiguous starting at
    `mem_base` (default 0).
- The head layer can reuse the previous membrane region when
    --reuse-last-mem-for-head is supplied (default True).
"""

from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
from pathlib import Path

import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from snn.models.spiking_flownet import EventSNNFlowNetLite


@dataclass
class LayerInfo:
    name: str
    in_c: int
    out_c: int
    kernel: int
    groups: int
    upscale_flag: bool
    downscale_flag: bool
    snn: bool
    h_in: int
    w_in: int
    h_out: int
    w_out: int
    num_weight_addr: int
    weight_addr: int
    num_mem_addr: int
    mem_addr: int


def align_up(value: int, align: int) -> int:
    if align <= 1:
        return value
    return ((value + align - 1) // align) * align


def conv_output(hw: Tuple[int, int], kernel: int, stride: int, padding: int) -> Tuple[int, int]:
    h, w = hw
    h_out = ((h + 2 * padding - kernel) // stride) + 1
    w_out = ((w + 2 * padding - kernel) // stride) + 1
    return h_out, w_out


def extract_conv_params(module) -> Tuple[int, int, int, int, int]:
    """Return (in_c, out_c, kernel, stride, padding, group    python tools/generate_instruct_mem.py \
      --config snn/configs/event_snn_lite.yaml \
      --output ../Main/rtl/src/mem/instruct.mem \
      --input-size 320 320 \
      --weight-base 0x0 --weight-align 16 --mem-pack 16 \
      --reuse-last-mem-for-heads) from a block."""
    conv = module.conv
    # ConvBlock/SpikingConvBlock wrap QuantizedConv2d at conv.conv
    if hasattr(conv, "conv"):
        conv = conv.conv
    k = conv.kernel_size[0]
    s = conv.stride[0]
    p = conv.padding[0]
    g = conv.groups
    return conv.in_channels, conv.out_channels, k, s, p, g


def gather_layers(
    model: EventSNNFlowNetLite,
    input_hw: Tuple[int, int],
    weight_base: int,
    weight_align: int,
    mem_base: int,
    mem_pack: int,
    reuse_last_mem_for_head: bool,
) -> List[LayerInfo]:
    # Execution order and spatial transforms between layers
    # Tuple: (name, module, upscale_flag, upsample_after, pool_after)
    ordered: List[Tuple[str, object, bool, bool, bool]] = [
        ("e1", model.e1, False, False, True),   # conv stride=1, then maxpool2d
        ("e2", model.e2, False, False, True),   # conv stride=1, then maxpool2d
        ("e3", model.e3, False, False, False),
        ("d3", model.d3, False, False, False),
        ("d2", model.d2, False, True, False),   # upsample after conv
        ("d1", model.d1, True, True, False),    # upsample after conv (mem stored pre-upsample)
        ("flow_head", model.flow_head, True, False, False),
    ]

    weight_ptr = weight_base
    mem_ptr = mem_base
    infos: List[LayerInfo] = []
    prev_mem_addr = None
    prev_num_mem = None

    h_cur, w_cur = input_hw

    for name, module, upscale_flag, upsample_after, pool_after in ordered:
        in_c, out_c, k, s, p, g = extract_conv_params(module)
        snn_layer = module.__class__.__name__.startswith("Spiking")

        # Convolution output (pre-pooling/upsampling)
        h_out, w_out = conv_output((h_cur, w_cur), k, s, p)

        weight_count = out_c * (in_c // g) * (k * k)
        num_weight_addr = math.ceil(weight_count / 9)
        weight_addr = align_up(weight_ptr, weight_align)
        weight_ptr = weight_addr + num_weight_addr

        if name == "e1":
            num_mem_addr = 0
            mem_addr = 0
        elif reuse_last_mem_for_head and name == "flow_head" and prev_mem_addr is not None:
            num_mem_addr = prev_num_mem
            mem_addr = prev_mem_addr
        else:
            num_mem_addr = math.ceil((h_out * w_out * out_c) / mem_pack)
            mem_addr = mem_ptr
            mem_ptr += num_mem_addr

        info = LayerInfo(
            name=name,
            in_c=in_c,
            out_c=out_c,
            kernel=k,
            groups=g,
            upscale_flag=upscale_flag,
            downscale_flag=pool_after,
            snn=snn_layer,
            h_in=h_cur,
            w_in=w_cur,
            h_out=h_out,
            w_out=w_out,
            num_weight_addr=num_weight_addr,
            weight_addr=weight_addr,
            num_mem_addr=num_mem_addr,
            mem_addr=mem_addr,
        )
        infos.append(info)

        prev_mem_addr = mem_addr
        prev_num_mem = num_mem_addr

        # Next-layer spatial size accounts for pooling/upsampling outside conv
        h_next, w_next = h_out, w_out
        if pool_after:
            h_next = h_next // 2
            w_next = w_next // 2
        if upsample_after:
            h_next *= 2
            w_next *= 2

        h_cur, w_cur = h_next, w_next

    return infos


def format_instr_lines(layers: Iterable[LayerInfo]) -> List[str]:
    lines: List[str] = [
        "# Reserved[19:0]  In_C[3:0] Out_C[3:0] Group[1:0] Upscale[0] Downscale[0] SNN[0]",
        "# H[8:0] W[8:0] Number of Weight addresses [8:0] Weight Address [8:0]",
        "# Reserved[6:0] Number of Membrane addresses [12:0] Membrane Potential Start Address [15:0]",
    ]

    for info in layers:
        lines.append(f"# {info.name} =========================")
        lines.append(
                "W {in_c} {out_c} {groups} {upscale} {downscale} {snn}".format(
                in_c=info.in_c,
                out_c=info.out_c,
                groups=info.groups,
                upscale=int(info.upscale_flag),
                downscale=int(info.downscale_flag),
                snn=int(info.snn),
            )
        )
        lines.append(
            "W {h} {w} {num_w} {addr:03X}".format(
                h=info.h_in,
                w=info.w_in,
                num_w=info.num_weight_addr,
                addr=info.weight_addr,
            )
        )
        lines.append(
            "W {num_mem} {addr:04X}".format(
                num_mem=info.num_mem_addr,
                addr=info.mem_addr,
            )
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate instruct.mem from EventSNNFlowNetLite")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--output", default="./instruct.mem", help="Destination .mem file")
    parser.add_argument("--input-size", nargs=2, type=int, metavar=("H", "W"), default=None,
                        help="Input height and width; defaults to config camera_size")
    parser.add_argument("--weight-base", type=lambda x: int(x, 0), default=0x0,
                        help="Starting weight address (hex or int)")
    parser.add_argument("--weight-align", type=int, default=16, help="Alignment for weight addresses")
    parser.add_argument("--mem-base", type=int, default=0, help="Starting membrane address")
    parser.add_argument("--mem-pack", type=int, default=16, help="Activations per membrane word")
    parser.add_argument("--reuse-last-mem-for-head", action="store_true", default=True,
                        help="Reuse previous layer membrane region for flow_head (mirrors sample)")
    parser.add_argument("--no-reuse-last-mem-for-head", dest="reuse_last_mem_for_head", action="store_false",
                        help="Disable membrane reuse for flow_head")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    input_hw = tuple(args.input_size) if args.input_size else tuple(config.get("camera_size", (320, 320)))
    if len(input_hw) != 2:
        raise ValueError("input size must have two elements (H, W)")

    model = EventSNNFlowNetLite(config)

    layers = gather_layers(
        model=model,
        input_hw=input_hw,
        weight_base=args.weight_base,
        weight_align=args.weight_align,
        mem_base=args.mem_base,
        mem_pack=args.mem_pack,
        reuse_last_mem_for_head=args.reuse_last_mem_for_head,
    )

    lines = format_instr_lines(layers)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
