import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any

class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator for quantization
    Forward: Quantize
    Backward: Pass gradient through unchanged
    """
    @staticmethod
    def forward(ctx, input):
        return input.round()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BinaryActivation(torch.autograd.Function):
    """
    Binary activation function with STE
    Forward: Sign function
    Backward: Straight-through or clipped gradient
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Clip gradient to [-1, 1] region
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1.0] = 0
        return grad_input


class BinaryWeight(torch.autograd.Function):
    """
    Binary weight quantization with scaling
    Uses mean absolute value as scaling factor
    """
    @staticmethod
    def forward(ctx, weight):
        # Calculate scaling factor (mean absolute value per filter)
        alpha = weight.abs().mean(dim=[1, 2, 3], keepdim=True)
        # Binarize
        binary_weight = torch.sign(weight)
        # Scale
        scaled_weight = alpha * binary_weight
        return scaled_weight
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class QuantizedWeight(torch.autograd.Function):
    """
    Multi-bit weight quantization with per-channel scaling
    
    Uses symmetric quantization with per-channel (per-filter) scaling
    for better accuracy compared to per-tensor scaling.
    """
    @staticmethod
    def forward(ctx, weight, bit_width, scale=None):
        # Per-channel (per-output-filter) quantization
        # Shape: [out_ch, in_ch, kH, kW] -> scale per out_ch

        # Symmetric quantization range
        qmax = 2 ** (bit_width - 1) - 1

        # Allow passing a precomputed per-channel scale (e.g., EMA/peak-tracked).
        if scale is None:
            # Calculate per-channel max absolute value
            w_reshaped = weight.abs().view(weight.size(0), -1)  # [out_ch, in_ch*kH*kW]
            max_abs = w_reshaped.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)  # [out_ch, 1, 1, 1]
            scale = max_abs / qmax

        scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero

        # Quantize then dequantize
        w_q = weight / scale
        w_q = torch.clamp(w_q, -qmax, qmax)
        w_q = w_q.round()
        return w_q * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None