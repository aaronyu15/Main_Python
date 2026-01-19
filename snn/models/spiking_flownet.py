"""
Spiking FlowNet for Optical Flow Estimation
Event-based/spike-based architecture for optical flow prediction
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .snn_layers import SpikingConvBlock
import torch.nn.functional as F

# ----------------------------
# Utilities: integer-friendly running mean
# ----------------------------
def running_mean_update(mean, x, t_idx: int, use_integer_approx: bool = False):
    """
    Online mean update: mean_t = mean_{t-1} + (x - mean_{t-1}) / (t_idx + 1)
    t_idx is 0-based.
    
    Args:
        mean: Previous running mean (None on first iteration)
        x: New value to incorporate
        t_idx: Current timestep index (0-based)
        use_integer_approx: Use bit-shift approximation instead of division
                           (hardware-friendly but less accurate)
    """
    if mean is None:
        return x
    
    if use_integer_approx:
        # Hardware-friendly approximation using bit shifts
        # Instead of dividing by (t+1), use power-of-2 approximation
        # This is less accurate but uses only shifts and adds
        # Find nearest power of 2: 2^floor(log2(t+1))
        import math
        shift = max(0, int(math.log2(t_idx + 1)))
        # mean_new ≈ mean + (x - mean) >> shift
        return mean + ((x - mean) / (2 ** shift))
    else:
        # Standard floating-point division
        return mean + (x - mean) / float(t_idx + 1)


# ----------------------------
# Recommended hardware-first model
# 1) Decoder runs per timestep (streaming-friendly)
# 2) Skip connections use ADD (not CONCAT) + 1x1 alignment
# 3) Online running means for skips (no large accumulators needed)
# 4) Flow scale is power-of-two by default (shift-friendly)
# ----------------------------
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
        tau=2.0,
        threshold=1.0,
        alpha=10.0,
        use_bn=False,
        quantize_weights=False,
        quantize_activations=False,
        quantize_mem=False,
        weight_bit_width=8,
        act_bit_width=8,
        binarize=False,
        flow_scale_pow2=4,  # 2^4 = 16 default (shift-friendly). set to None to disable.
        return_last_flow=True,
        hardware_mode=False,  # Enable hardware-friendly integer approximations
        output_bit_width=16,  # Bit width for output layer (higher precision for flow)
        first_layer_bit_width=8,  # Bit width for first encoder layer (higher precision helps)
        mem_bit_width=16,  # Bit width for membrane potential quantization
        enable_logging=False,  # Enable detailed statistics logging
        logger=None,  # TensorBoard logger instance
    ):
        super().__init__()
        
        # Binarize overrides weight/act bit widths to 1
        if binarize:
            weight_bit_width = 1
            act_bit_width = 1
        
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.quantize_mem = quantize_mem
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width
        self.return_last_flow = return_last_flow
        self.hardware_mode = hardware_mode  # For bit-exact hardware matching
        self.output_bit_width = output_bit_width  # Higher precision for output layer
        self.first_layer_bit_width = first_layer_bit_width  # Higher precision for first layer
        self.mem_bit_width = mem_bit_width  # Membrane potential quantization
        self.enable_logging = enable_logging  # For debugging quantization
        self.logger = logger  # TensorBoard logger

        # Encoder - First layer uses higher precision for better input processing
        self.e1 = SpikingConvBlock(
            2, base_ch, k=5, s=2, p=2,
            tau=tau, threshold=threshold, alpha=alpha,
            use_bn=use_bn,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            quantize_mem=quantize_mem,
            weight_bit_width=first_layer_bit_width,
            act_bit_width=first_layer_bit_width,
            hardware_mode=hardware_mode, mem_bit_width=self.mem_bit_width,
            enable_logging=enable_logging, layer_name="e1", logger=logger
        )
        self.e2 = SpikingConvBlock(
            base_ch, base_ch * 2, k=3, s=2, p=1,
            tau=tau, threshold=threshold, alpha=alpha,
            use_bn=use_bn,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            quantize_mem=quantize_mem,
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            hardware_mode=hardware_mode, mem_bit_width=self.mem_bit_width,
            enable_logging=enable_logging, layer_name="e2", logger=logger
        )
        self.e3 = SpikingConvBlock(
            base_ch * 2, base_ch * 4, k=3, s=2, p=1,
            tau=tau, threshold=threshold, alpha=alpha,
            use_bn=use_bn,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            quantize_mem=quantize_mem,
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            hardware_mode=hardware_mode, mem_bit_width=self.mem_bit_width,
            enable_logging=enable_logging, layer_name="e3", logger=logger
        )

        # Decoder blocks (note: channels stay consistent because skips are ADD, not CONCAT)
        self.d3 = SpikingConvBlock(
            base_ch * 4, base_ch * 2, k=3, s=1, p=1,
            tau=tau, threshold=threshold, alpha=alpha,
            use_bn=use_bn,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            quantize_mem=quantize_mem,
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            hardware_mode=hardware_mode, mem_bit_width=self.mem_bit_width,
            enable_logging=enable_logging, layer_name="d3", logger=logger
        )
        self.d2 = SpikingConvBlock(
            base_ch * 2, base_ch, k=3, s=1, p=1,
            tau=tau, threshold=threshold, alpha=alpha,
            use_bn=use_bn,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            quantize_mem=quantize_mem,
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            hardware_mode=hardware_mode, mem_bit_width=self.mem_bit_width,
            enable_logging=enable_logging, layer_name="d2", logger=logger
        )
        self.d1 = SpikingConvBlock(
            base_ch, base_ch, k=3, s=1, p=1,
            tau=tau, threshold=threshold, alpha=alpha,
            use_bn=use_bn,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            quantize_mem=quantize_mem,
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            hardware_mode=hardware_mode, mem_bit_width=self.mem_bit_width,
            enable_logging=enable_logging, layer_name="d1", logger=logger
        )

        # 1x1 alignment convs for ADD skips (cheap, great for hardware)
        # Align skip feature channels to decoder channels at each scale.
        # Always use QuantizedConv2d for consistent structure (quantization can be disabled)
        from ..quantization import QuantizedConv2d
        self.skip2_align = QuantizedConv2d(
            base_ch * 2, base_ch * 2, kernel_size=1, bias=False,
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            enable_logging=enable_logging,
            layer_name="skip2_align",
            logger=logger
        )
        self.skip1_align = QuantizedConv2d(
            base_ch, base_ch, kernel_size=1, bias=False,
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            enable_logging=enable_logging,
            layer_name="skip1_align",
            logger=logger
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
        self.flow_scale_pow2 = flow_scale_pow2

        # Optional: if you later want fully-integer export, you can replace this
        # with a fixed-point scaling strategy at export time.

    def _apply_flow_scale(self, flow):
        if self.flow_scale_pow2 is None:
            return flow
        # multiply by 2^k
        return flow * (2.0 ** int(self.flow_scale_pow2))

    def forward(self, x):
        # Accept packed or [N,T,2,H,W]
        if x.dim() == 4:
            N, CT, H, W = x.shape
            assert CT % 2 == 0, "Expected channels = 2*T"
            T = CT // 2
            x = x.view(N, T, 2, H, W)
        else:
            N, T, C, H, W = x.shape
            assert C == 2, "Expected 2 polarity channels"

        # Encoder membrane states (these are the main temporal states you keep)
        mem_e1 = mem_e2 = mem_e3 = None

        # Decoder membrane states (now meaningful because we decode every timestep)
        mem_d3 = mem_d2 = mem_d1 = None

        # Running means for skip features (online rate coding)
        mean_e1 = None  # [N, base,   H/2, W/2]
        mean_e2 = None  # [N, 2base,  H/4, W/4]

        # Flow aggregation options:
        flow_mean = None

        for t in range(T):
            xt = x[:, t]  # [N,2,H,W]

            # --- Encode timestep ---
            s1, mem_e1 = self.e1(xt, mem_e1)   # [N, base,   H/2, W/2]
            s2, mem_e2 = self.e2(s1, mem_e2)   # [N, 2base,  H/4, W/4]
            s3, mem_e3 = self.e3(s2, mem_e3)   # [N, 4base,  H/8, W/8]

            # Update online means for skips (instead of accumulating all T then dividing)
            # Use hardware_mode for integer approximation when needed
            mean_e1 = running_mean_update(mean_e1, s1, t, use_integer_approx=self.hardware_mode)
            mean_e2 = running_mean_update(mean_e2, s2, t, use_integer_approx=self.hardware_mode)

            # --- Decode timestep (streaming-friendly) ---
            d3 = F.interpolate(s3, scale_factor=2, mode="nearest")  # -> H/4
            d3, mem_d3 = self.d3(d3, mem_d3)

            # ADD skip at H/4: align mean_e2 then add
            skip2 = self.skip2_align(mean_e2)
            d3 = d3 + skip2

            d2 = F.interpolate(d3, scale_factor=2, mode="nearest")  # -> H/2
            d2, mem_d2 = self.d2(d2, mem_d2)

            # ADD skip at H/2: align mean_e1 then add
            skip1 = self.skip1_align(mean_e1)
            d2 = d2 + skip1

            d1 = F.interpolate(d2, scale_factor=2, mode="nearest")  # -> H
            d1, mem_d1 = self.d1(d1, mem_d1)

            flow_t = self._apply_flow_scale(self.flow_head(d1))

            flow_mean = running_mean_update(flow_mean, flow_t, t, use_integer_approx=self.hardware_mode)

            flow_last = flow_t  # keep last flow for option

        if self.return_last_flow:
            # Option A: return last flow (simple) -> just keep overwriting
            return {"flow": flow_last}
        else:
            # Option B: return mean flow over time (stable)
            return {"flow": flow_mean}