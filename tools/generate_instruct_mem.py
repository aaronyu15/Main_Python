"""
Generate hardware instruction memory files (instruct.mem) from EventSNNFlowNetLite.

Walks model layers in order, derives convolution parameters, estimates output
feature map sizes, and allocates weight/membrane address ranges.

Example:
    python tools/generate_instruct_mem.py \
        --config snn/configs/event_snn_lite.yaml \
        --output ../Main/rtl/src/mem/instruct.mem \
        --input-size 320
"""

from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Tuple
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
    stride: int
    snn: bool
    dim: int
    num_weight_addr: int
    weight_addr: int
    num_mem_addr: int
    mem_addr: int


def align_up(value: int, align: int) -> int:
    if align <= 1:
        return value
    return ((value + align - 1) // align) * align


def extract_conv_params(module) -> Tuple[int, int, int, int, int, int]:
    """Return (in_c, out_c, kernel, stride, padding, groups) from a block."""
    conv = module.conv
    if hasattr(conv, "conv"):
        conv = conv.conv
    return (
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size[0],
        conv.stride[0],
        conv.padding[0],
        conv.groups,
    )


def gather_layers(
    model: EventSNNFlowNetLite,
    input_dim: int,
    weight_base: int,
    weight_align: int,
    mem_base: int,
    mem_pack: int,
) -> List[LayerInfo]:
    """
    Build layer info list based on forward pass:
    E1(s=2) -> E2(s=2) -> E3(s=2) -> E4(s=1) -> D4(s=1) -> D3(s=1) -> D2(s=1) -> D1(s=1)
    Then D1 upsamples 4x (160), then 2x more to 320 for flow_head
    """
    layers_config = [
        ("e1", model.e1),
        ("e2", model.e2),
        ("e3", model.e3),
        ("e4", model.e4),
        ("d4", model.d4),
        ("d3", model.d3),
        ("d2", model.d2),
        ("flow_head", model.flow_head),
    ]

    weight_ptr = weight_base
    mem_ptr = mem_base
    infos: List[LayerInfo] = []

    dim_cur = input_dim

    for name, module in layers_config:
        in_c, out_c, k, s, p, g = extract_conv_params(module)
        snn_layer = module.__class__.__name__.startswith("Spiking")

        # flow_head receives upsampled input (back to full resolution)
        if name == "flow_head":
            dim_cur = input_dim

        # Conv output dim
        dim_out = ((dim_cur + 2 * p - k) // s) + 1

        # Weight allocation: 9 weights per memory word
        weight_count = out_c * (in_c // g) * (k * k)
        num_weight_addr = math.ceil(weight_count / 9)
        weight_addr = align_up(weight_ptr, weight_align)
        weight_ptr = weight_addr + num_weight_addr

        # Membrane allocation
        if name == "e1" or name == "e2":
            # No membrane for spike_no_membrane layers
            num_mem_addr = 0
            mem_addr = 0
        elif name == "flow_head":
            # flow_head is not SNN, no membrane
            num_mem_addr = 0
            mem_addr = 0
        else:
            num_mem_addr = math.ceil((dim_out * dim_out) / mem_pack)
            mem_addr = mem_ptr
            mem_ptr += num_mem_addr

        info = LayerInfo(
            name=name,
            in_c=in_c,
            out_c=out_c,
            stride=s,
            snn=snn_layer,
            dim=dim_cur,
            num_weight_addr=num_weight_addr,
            weight_addr=weight_addr,
            num_mem_addr=num_mem_addr,
            mem_addr=mem_addr,
        )
        infos.append(info)

        dim_cur = dim_out

    return infos


def pack_instruction(info: LayerInfo) -> List[int]:
    """
    Pack layer info into 3 32-bit instruction words (little endian):
    - Word 0: [reserved(21) | in_c(6) | out_c(6) | stride(2) | snn(1)]
    - Word 1: [reserved(2) | dim(9) | num_weight_addr(9) | weight_addr(12)]
    - Word 2: [reserved(8) | num_mem_addr(12) | mem_addr(14)]
    Returns list of 3 ints.
    """
    # Word 0: bits 31:11 reserved, 10:7 in_c, 6:3 out_c, 2:1 stride, 0 snn
    word0 = (
        ((info.in_c & 0x3F) << 9) |      # bits 10:7
        ((info.out_c & 0x3F) << 3) |     # bits 6:3
        ((info.stride & 0x3) << 1) |    # bits 2:1
        ((int(info.snn) & 0x1) << 0)    # bit 0
        # reserved upper bits (31:11) are 0
    )
    # Word 1: bits 31:30 reserved, 29:21 dim, 20:12 num_weight_addr, 11:0 weight_addr
    word1 = (
        ((info.dim & 0x1FF) << 21) |             # bits 29:21
        ((info.num_weight_addr & 0x1FF) << 12) | # bits 20:12
        ((info.weight_addr & 0xFFF) << 0)        # bits 11:0
        # reserved upper bits (31:30) are 0
    )
    # Word 2: bits 31:24 reserved, 23:12 num_mem_addr, 11:0 mem_addr
    word2 = (
        ((info.num_mem_addr & 0xFFF) << 14) |    # bits 23:12
        ((info.mem_addr & 0x3FFF) << 0)           # bits 11:0
        # reserved upper bits (31:24) are 0
    )
    return [word0, word1, word2]


def format_instruction_text(info: LayerInfo, words: List[int]) -> List[str]:
    """Return human-readable lines for each instruction word."""
    lines = [
        f"{info.name}:",
        f"  Word 0: 0x{words[0]:08X} | [reserved(21) | in_c={info.in_c} | out_c={info.out_c} | stride={info.stride} | snn={int(info.snn)}]",
        f"  Word 1: 0x{words[1]:08X} | [reserved(2) | dim={info.dim} | num_weight_addr={info.num_weight_addr} | weight_addr={info.weight_addr}]",
        f"  Word 2: 0x{words[2]:08X} | [reserved(8) | num_mem_addr={info.num_mem_addr} | mem_addr={info.mem_addr}]",
        ""
    ]
    return lines


def format_instr_lines(layers: Iterable[LayerInfo]) -> List[str]:
    lines: List[str] = [
        "# In_C[3:0] Out_C[3:0] Stride[1:0] SNN[0]",
        "# dim[8:0] Number of Weight addresses [8:0] Weight Address [8:0]",
        "# Number of Membrane addresses [12:0] Membrane Potential Start Address [15:0]",
    ]

    for info in layers:
        lines.append(f"# {info.name} =========================")
        lines.append(
            "W {in_c} {out_c} {stride} {snn}".format(
                in_c=info.in_c,
                out_c=info.out_c,
                stride=info.stride,
                snn=int(info.snn),
            )
        )
        lines.append(
            "W {dim} {num_w} {addr:03X}".format(
                dim=info.dim,
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
    parser.add_argument("--input-size", type=int, default=None,
                        help="Input dimension (square); defaults to config camera_size[0]")
    parser.add_argument("--weight-base", type=lambda x: int(x, 0), default=0x0,
                        help="Starting weight address (hex or int)")
    parser.add_argument("--weight-align", type=int, default=1, help="Alignment for weight addresses")
    parser.add_argument("--mem-base", type=int, default=0, help="Starting membrane address")
    parser.add_argument("--mem-pack", type=int, default=1, help="Activations per membrane word")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    input_dim = args.input_size if args.input_size else config.get("camera_size", [320, 320])[0]

    model = EventSNNFlowNetLite(config)

    layers = gather_layers(
        model=model,
        input_dim=input_dim,
        weight_base=args.weight_base,
        weight_align=args.weight_align,
        mem_base=args.mem_base,
        mem_pack=args.mem_pack,
    )

    # Generate .mem and .txt
    mem_lines: List[str] = []
    txt_lines: List[str] = []
    for info in layers:
        words = pack_instruction(info)
        for w in words:
            mem_lines.append(f"{w:08X}")
        txt_lines.extend(format_instruction_text(info, words))

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(mem_lines) + "\n", encoding="ascii")

    txt_path = out_path.with_suffix('.txt')
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"Wrote {len(mem_lines)} lines to {out_path}")
    print(f"Wrote human-readable instructions to {txt_path}")


if __name__ == "__main__":
    main()
