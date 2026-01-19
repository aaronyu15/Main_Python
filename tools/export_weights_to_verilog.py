#!/usr/bin/env python3

"""Export quantized EventSNNFlowNetLite weights for SystemVerilog.

Python-side QAT in this repo uses *fake quantization*:
- Weights: symmetric quantization in the forward pass (see QuantizedWeight)
- Activations: symmetric fake-quant via EMA min/max (see QuantizationAwareLayer)

For RTL deployment we need frozen integers. This script exports:
- int8 weights per conv, flattened as [out][in][ky][kx] for $readmemh
- int32 biases for the biased spiking conv blocks (e1/e2/e3/d3/d2/d1)
- a SystemVerilog include file with suggested per-layer shift parameters,
  plus DECAY_Q and THRESH_Q for the LIF dynamics

Important: This exporter intentionally uses a *power-of-two per-layer scale*
(i.e. requantization by shifts), which is convenient for RTL but is an
approximation of the Python per-output-channel quantizer.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml


def _load_yaml(path: Path) -> Dict:
	with path.open("r") as f:
		return yaml.safe_load(f)


def _load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
	ckpt = torch.load(path, map_location="cpu")
	if isinstance(ckpt, dict):
		for key in ("model_state_dict", "state_dict", "model"):
			if key in ckpt and isinstance(ckpt[key], dict):
				return ckpt[key]
		# Sometimes the checkpoint is already a state_dict-like mapping.
		if all(isinstance(k, str) for k in ckpt.keys()):
			return ckpt  # type: ignore[return-value]
	raise ValueError(f"Unrecognized checkpoint format: {path}")


def _twos_comp_hex(value: int, bits: int) -> str:
	mask = (1 << bits) - 1
	return f"{value & mask:0{bits // 4}x}"


def _quantize_weight_pow2_per_layer(
	w_fp: torch.Tensor, bit_width: int
) -> Tuple[torch.Tensor, float, int]:
	"""Quantize weights with a power-of-two *per-layer* scale.

	Returns:
	  - w_q: int8 tensor, where approximately w_fp ~= w_q * scale_pow2
	  - scale_pow2: float power-of-two scale (2^{-shift_s})
	  - shift_s: integer such that scale_pow2 = 2^{-shift_s}
	"""
	if bit_width < 2:
		raise ValueError("bit_width must be >=2")

	qmax = (2 ** (bit_width - 1)) - 1
	max_abs = float(w_fp.abs().max().item())

	scale_fp = max_abs / float(qmax) if max_abs > 0.0 else 1.0
	if not math.isfinite(scale_fp) or scale_fp <= 0.0:
		scale_fp = 1.0

	# Choose power-of-two scale ~ scale_fp
	shift_s = int(round(-math.log2(scale_fp)))
	shift_s = max(0, min(31, shift_s))
	scale_pow2 = 2.0 ** (-shift_s)

	w_q = torch.round(w_fp / scale_pow2).clamp(-qmax, qmax).to(torch.int8)
	return w_q, scale_pow2, shift_s


def _export_int8_memh(path: Path, values: torch.Tensor) -> None:
	values_i8 = values.to(torch.int8).flatten().tolist()
	with path.open("w") as f:
		for v in values_i8:
			f.write(_twos_comp_hex(int(v), 8) + "\n")


def _export_int32_memh(path: Path, values: torch.Tensor) -> None:
	values_i32 = values.to(torch.int32).flatten().tolist()
	with path.open("w") as f:
		for v in values_i32:
			f.write(_twos_comp_hex(int(v), 32) + "\n")


def _compute_decay_q(tau: float, mem_q: int) -> int:
	decay = math.exp(-1.0 / float(tau))
	return int(round(decay * (2**mem_q)))


def _compute_thresh_q(threshold: float, mem_q: int) -> int:
	return int(round(float(threshold) * (2**mem_q)))


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--config", type=str, required=True)
	ap.add_argument("--checkpoint", type=str, required=True)
	ap.add_argument("--out-dir", type=str, required=True)
	ap.add_argument("--bit-width", type=int, default=None, help="Override bit width")
	ap.add_argument("--mem-q", type=int, default=8, help="Fraction bits for membrane/current")
	args = ap.parse_args()

	config_path = Path(args.config)
	ckpt_path = Path(args.checkpoint)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	cfg = _load_yaml(config_path)

	bit_width = int(args.bit_width) if args.bit_width is not None else int(cfg.get("initial_bit_width", 8))
	if bit_width < 2:
		raise ValueError("This exporter currently supports multi-bit (>=2) quantization only.")

	# Build model using the same helper as training.
	# NOTE: This uses cwd-relative import; run from Main_Python.
	from train import build_model

	model = build_model(cfg)
	state = _load_checkpoint_state(ckpt_path)
	missing, unexpected = model.load_state_dict(state, strict=False)
	if missing:
		print("[export] Missing keys:", missing)
	if unexpected:
		print("[export] Unexpected keys:", unexpected)
	model.eval()

	convs = {
		"e1": model.e1.conv.conv,
		"e2": model.e2.conv.conv,
		"e3": model.e3.conv.conv,
		"d3": model.d3.conv.conv,
		"d2": model.d2.conv.conv,
		"d1": model.d1.conv.conv,
		"skip2": model.skip2_align.conv,
		"skip1": model.skip1_align.conv,
		"flow": model.flow_head.conv,
	}

	shifts: Dict[str, int] = {}
	scales: Dict[str, float] = {}

	for name, conv in convs.items():
		w_fp = conv.weight.detach().cpu()
		w_q, w_scale_pow2, shift_s = _quantize_weight_pow2_per_layer(w_fp, bit_width)

		# Reference RTL uses: cur_q = (acc << MEM_Q) >> LAYER_SHIFT
		# If weights are represented as w_fp ~= w_q * 2^{-s}, then conv output ~= 2^{-s} * acc
		# => choose LAYER_SHIFT = MEM_Q + s.
		layer_shift = int(args.mem_q) + shift_s
		shifts[name] = layer_shift
		scales[name] = w_scale_pow2

		_export_int8_memh(out_dir / f"w_{name}.hex", w_q)

		if conv.bias is not None and name in {"e1", "e2", "e3", "d3", "d2", "d1"}:
			b_fp = conv.bias.detach().cpu()
			b_q = torch.round(b_fp / w_scale_pow2).to(torch.int32)
			_export_int32_memh(out_dir / f"b_{name}.hex", b_q)

	# Emit SV include in Main_Verilog/rtl so it can be `include'd by RTL.
	rtl_inc = Path(__file__).resolve().parents[2] / "Main_Verilog" / "rtl" / "eventsnn_flownet_lite_v2_params.svh"
	rtl_inc.parent.mkdir(parents=True, exist_ok=True)

	tau = float(cfg.get("tau", 2.0))
	threshold = float(cfg.get("threshold", 1.0))
	mem_q = int(args.mem_q)

	decay_q = _compute_decay_q(tau, mem_q)
	thresh_q = _compute_thresh_q(threshold, mem_q)
	flow_scale_pow2 = int(cfg.get("flow_scale_pow2", 4)) if "flow_scale_pow2" in cfg else 4

	with rtl_inc.open("w") as f:
		f.write("// Auto-generated by Main_Python/tools/export_weights_to_verilog.py\n")
		f.write("// Re-run the exporter to regenerate.\n\n")
		f.write("`ifndef EVENTSNN_FLOWNET_LITE_V2_PARAMS_SVH\n")
		f.write("`define EVENTSNN_FLOWNET_LITE_V2_PARAMS_SVH\n\n")
		f.write(f"`define EVENTSNN_MEM_Q {mem_q}\n")
		f.write(f"`define EVENTSNN_DECAY_Q {decay_q}\n")
		f.write(f"`define EVENTSNN_THRESH_Q {thresh_q}\n\n")
		f.write(f"`define EVENTSNN_FLOW_SCALE_POW2 {flow_scale_pow2}\n\n")
		f.write(f"`define EVENTSNN_E1_SHIFT {shifts['e1']}\n")
		f.write(f"`define EVENTSNN_E2_SHIFT {shifts['e2']}\n")
		f.write(f"`define EVENTSNN_E3_SHIFT {shifts['e3']}\n")
		f.write(f"`define EVENTSNN_D3_SHIFT {shifts['d3']}\n")
		f.write(f"`define EVENTSNN_D2_SHIFT {shifts['d2']}\n")
		f.write(f"`define EVENTSNN_D1_SHIFT {shifts['d1']}\n")
		f.write(f"`define EVENTSNN_SKIP2_SHIFT {shifts['skip2']}\n")
		f.write(f"`define EVENTSNN_SKIP1_SHIFT {shifts['skip1']}\n")
		f.write(f"`define EVENTSNN_FLOW_SHIFT {shifts['flow']}\n\n")
		f.write("`endif\n")

	print("[export] Wrote weights to:", out_dir)
	print("[export] Wrote SV include:", rtl_inc)
	for name in ("e1", "e2", "e3", "d3", "d2", "d1", "skip2", "skip1", "flow"):
		print(f"[export] {name}: scale≈{scales[name]:.6g}, shift={shifts[name]}")


if __name__ == "__main__":
	main()