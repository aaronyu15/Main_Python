"""
Spiking FlowNet for Optical Flow Estimation
Event-based/spike-based architecture for optical flow prediction
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .snn_layers import SpikingConvBlock
from ..quantization import QuantizedConv2d
import torch.nn.functional as F


def skip_update(state, x, shift_k: int):
    """
    state <- state + (x - state) / 2^k   (hardware: add + sub + >>k)
    If shift_k=0 => state=x immediately.
    """
    if state is None:
        return x
    if shift_k == 0:
        return x
    return state + x 


class EventSNNFlowNetLite(nn.Module):
    """
    Input:  x [N, T, 2, H, W]  (T time bins, 2 polarities)
        OR   x [N, 2*T, H, W]  (packed)

    Output: dict(flow=[N, 2, H, W])

    Notes:
    - This version decodes every timestep, so you can later map it to a streaming
      pipeline more naturally.
    - Skip connections are ADD-based for bandwidth/memory savings.
    - Keeps only small membrane states + running means, not full T accumulators.
    """
    def __init__(
        self,
        base_ch=32,
        decay=0.5,
        threshold=1.0,
        alpha=10.0,
        quantize_weights=False,
        quantize_activations=False,
        quantize_mem=False,
        weight_bit_width=8,
        act_bit_width=8,
        output_bit_width=16,  # Bit width for output layer (higher precision for flow)
        first_layer_bit_width=8,  # Bit width for first encoder layer (higher precision helps)
        mem_bit_width=16,  # Bit width for membrane potential quantization
        enable_logging=False,  # Enable detailed statistics logging
        logger=None,  # TensorBoard logger instance
    ):
        super().__init__()
        
        
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.quantize_mem = quantize_mem
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width
        self.mem_bit_width = mem_bit_width  # Membrane potential quantization
        self.output_bit_width = output_bit_width  # Higher precision for output layer
        self.first_layer_bit_width = first_layer_bit_width  # Higher precision for first layer
        self.enable_logging = enable_logging  # For debugging quantization
        self.logger = logger  # TensorBoard logger

        # Encoder - First layer uses higher precision for better input processing
        self.e1 = SpikingConvBlock(
            2, base_ch, k=5, s=2, p=2,
            decay=decay, threshold=threshold, alpha=alpha,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            quantize_mem=quantize_mem,
            weight_bit_width=first_layer_bit_width,
            act_bit_width=first_layer_bit_width,
            mem_bit_width=self.mem_bit_width,
            enable_logging=enable_logging, layer_name="e1", logger=logger
        )
        self.e2 = SpikingConvBlock(
            base_ch, base_ch * 2, k=3, s=2, p=1,
            decay=decay, threshold=threshold, alpha=alpha,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            quantize_mem=quantize_mem,
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            mem_bit_width=self.mem_bit_width,
            enable_logging=enable_logging, layer_name="e2", logger=logger
        )
        self.e3 = SpikingConvBlock(
            base_ch * 2, base_ch * 4, k=3, s=2, p=1,
            decay=decay, threshold=threshold, alpha=alpha,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            quantize_mem=quantize_mem,
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            mem_bit_width=self.mem_bit_width,
            enable_logging=enable_logging, layer_name="e3", logger=logger
        )

        # Decoder blocks (note: channels stay consistent because skips are ADD, not CONCAT)
        self.d3 = SpikingConvBlock(
            base_ch * 4, base_ch * 2, k=3, s=1, p=1,
            decay=decay, threshold=threshold, alpha=alpha,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            quantize_mem=quantize_mem,
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            mem_bit_width=self.mem_bit_width,
            enable_logging=enable_logging, layer_name="d3", logger=logger
        )
        self.d2 = SpikingConvBlock(
            base_ch * 2, base_ch, k=3, s=1, p=1,
            decay=decay, threshold=threshold, alpha=alpha,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            quantize_mem=quantize_mem,
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            mem_bit_width=self.mem_bit_width,
            enable_logging=enable_logging, layer_name="d2", logger=logger
        )
        self.d1 = SpikingConvBlock(
            base_ch, base_ch, k=3, s=1, p=1,
            decay=decay, threshold=threshold, alpha=alpha,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            quantize_mem=quantize_mem,
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            mem_bit_width=self.mem_bit_width,
            enable_logging=enable_logging, layer_name="d1", logger=logger
        )

        # Flow head - final prediction layer
        # Uses higher precision (output_bit_width) than rest of network
        # This preserves output accuracy for low-bit quantization
        # Quantization is enabled to avoid floating point values
        self.flow_head = QuantizedConv2d(
            base_ch, 2, kernel_size=3, padding=1,
            weight_bit_width=self.output_bit_width,
            act_bit_width=self.output_bit_width,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            enable_logging=enable_logging,
            layer_name="flow_head",
            logger=logger
        )

        # Power-of-two scaling (shift-friendly); if None, scale=1
        self.log_flow_scale = nn.Parameter(torch.tensor(0.0))

        # Optional: if you later want fully-integer export, you can replace this
        # with a fixed-point scaling strategy at export time.

    def _apply_flow_scale(self, flow):
        if self.log_flow_scale is None:
            return flow
        # Clamp log_flow_scale to prevent overflow in exp (exp(10) ≈ 22000, exp(20) ≈ 5e8)
        log_scale_clamped = torch.clamp(self.log_flow_scale, min=-10.0, max=10.0)
        scale = torch.exp(log_scale_clamped)
        return flow * scale

    def forward(self, x):

        N, T, C, H, W = x.shape
        assert C == 2, "Expected 2 polarity channels"

        mem_e1 = mem_e2 = mem_e3 = None
        mem_d3 = mem_d2 = mem_d1 = None

        # Flow aggregation options:
        flow_acc = None  # [N,2,H,W]
        flow_last = None

        for t in range(T):
            xt = x[:, t]  # [N,2,H,W]

            # --- Encode timestep ---
            s1, mem_e1 = self.e1(xt, mem_e1)   # [N, base,   H/2, W/2]
            s2, mem_e2 = self.e2(s1, mem_e2)   # [N, 2base,  H/4, W/4]
            s3, mem_e3 = self.e3(s2, mem_e3)   # [N, 4base,  H/8, W/8]

            # --- Decode timestep (streaming-friendly) ---
            d3 = F.interpolate(s3, scale_factor=2, mode="nearest")  # -> H/4
            d3, mem_d3 = self.d3(d3, mem_d3)
            d3 = d3 + s2

            d2 = F.interpolate(d3, scale_factor=2, mode="nearest")  # -> H/2
            d2, mem_d2 = self.d2(d2, mem_d2)
            d2 = d2 + s1

            d1 = F.interpolate(d2, scale_factor=2, mode="nearest")  # -> H
            d1, mem_d1 = self.d1(d1, mem_d1)

            dflow_t = self._apply_flow_scale(self.flow_head(mem_d1))

            if flow_acc is None:
                flow_acc = dflow_t
            else:
                flow_acc = flow_acc + dflow_t

            flow_last = flow_acc

        return {"flow": flow_last}