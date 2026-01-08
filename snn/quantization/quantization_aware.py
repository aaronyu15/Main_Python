"""
Quantization-Aware Training for SNNs
Supports variable bit-width quantization with hardware deployment in mind
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


class QuantizationAwareLayer(nn.Module):
    """
    Quantization-aware layer that can be inserted into the network
    
    Args:
        bit_width: Number of bits for quantization (1 for binary)
        symmetric: Use symmetric quantization around zero
        per_channel: Quantize per channel vs per tensor
    """
    def __init__(
        self,
        bit_width: int = 8,
        symmetric: bool = True,
        per_channel: bool = False
    ):
        super().__init__()
        
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.per_channel = per_channel
        
        # Quantization levels
        if symmetric:
            self.qmin = -(2 ** (bit_width - 1))
            self.qmax = 2 ** (bit_width - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bit_width - 1
        
        # Learnable scale and zero-point
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.calibrated = False
        
    def calibrate(self, x: torch.Tensor):
        """Calibrate quantization parameters based on input statistics"""
        if self.calibrated:
            return
        
        with torch.no_grad():
            if self.symmetric:
                max_val = torch.max(torch.abs(x))
                self.scale = max_val / (2 ** (self.bit_width - 1) - 1)
                self.zero_point = torch.tensor(0.0, device=x.device)
            else:
                min_val = torch.min(x)
                max_val = torch.max(x)
                self.scale = (max_val - min_val) / (2 ** self.bit_width - 1)
                self.zero_point = min_val
        
        self.calibrated = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization-aware forward pass"""
        
        # Calibrate on first pass
        if not self.calibrated and self.training:
            self.calibrate(x)
        
        # Quantize
        if self.scale > 0:
            x_q = (x - self.zero_point) / self.scale
            x_q = torch.clamp(x_q, self.qmin, self.qmax)
            x_q = StraightThroughEstimator.apply(x_q)
            x_dequant = x_q * self.scale + self.zero_point
        else:
            x_dequant = x
        
        return x_dequant


class BinaryQuantizer(nn.Module):
    """
    Binary quantizer: quantizes to {-1, +1}
    Essential for binarized neural networks
    """
    def __init__(self, deterministic: bool = True):
        super().__init__()
        self.deterministic = deterministic
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Binarize input to {-1, +1}"""
        if self.deterministic:
            # Deterministic binarization
            return BinaryActivation.apply(x)
        else:
            # Stochastic binarization
            prob = (x + 1) / 2  # Map to [0, 1]
            binary = torch.bernoulli(prob)
            return 2 * binary - 1  # Map to {-1, +1}


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


def quantize_weights(weights: torch.Tensor, bit_width: int = 8) -> torch.Tensor:
    """
    Quantize weight tensor
    
    Args:
        weights: Weight tensor to quantize
        bit_width: Number of bits (1 for binary)
        
    Returns:
        Quantized weights
    """
    if bit_width == 1:
        # Binary weights
        return BinaryWeight.apply(weights)
    else:
        # Multi-bit quantization
        qmin = -(2 ** (bit_width - 1))
        qmax = 2 ** (bit_width - 1) - 1
        
        scale = torch.max(torch.abs(weights)) / (2 ** (bit_width - 1) - 1)
        
        if scale > 0:
            w_q = weights / scale
            w_q = torch.clamp(w_q, qmin, qmax)
            w_q = StraightThroughEstimator.apply(w_q)
            w_dequant = w_q * scale
            return w_dequant
        else:
            return weights


def quantize_activations(activations: torch.Tensor, bit_width: int = 8) -> torch.Tensor:
    """
    Quantize activation tensor
    
    Args:
        activations: Activation tensor to quantize
        bit_width: Number of bits (1 for binary)
        
    Returns:
        Quantized activations
    """
    if bit_width == 1:
        # Binary activations
        return BinaryActivation.apply(activations)
    else:
        # Multi-bit quantization
        qmin = 0
        qmax = 2 ** bit_width - 1
        
        min_val = torch.min(activations)
        max_val = torch.max(activations)
        scale = (max_val - min_val) / (2 ** bit_width - 1)
        
        if scale > 0:
            a_q = (activations - min_val) / scale
            a_q = torch.clamp(a_q, qmin, qmax)
            a_q = StraightThroughEstimator.apply(a_q)
            a_dequant = a_q * scale + min_val
            return a_dequant
        else:
            return activations


class QuantizedConv2d(nn.Conv2d):
    """
    Quantized Conv2d layer
    Automatically quantizes weights during forward pass
    """
    def __init__(self, *args, bit_width: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.bit_width = bit_width
    
    def forward(self, input):
        # Quantize weights
        if self.training or self.bit_width < 32:
            quantized_weight = quantize_weights(self.weight, self.bit_width)
        else:
            quantized_weight = self.weight
        
        # Standard conv2d operation with quantized weights
        return F.conv2d(
            input, quantized_weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


class PACT(nn.Module):
    """
    Parameterized Clipping Activation (PACT)
    Learnable clipping threshold for better quantization
    
    Paper: "PACT: Parameterized Clipping Activation for Quantized Neural Networks"
    """
    def __init__(self, init_clip: float = 6.0, bit_width: int = 8):
        super().__init__()
        self.clip_val = nn.Parameter(torch.tensor(init_clip))
        self.bit_width = bit_width
    
    def forward(self, x):
        # Clip to learnable range
        x_clipped = torch.clamp(x, 0, self.clip_val)
        
        # Quantize
        if self.training:
            scale = self.clip_val / (2 ** self.bit_width - 1)
            x_q = x_clipped / scale
            x_q = StraightThroughEstimator.apply(x_q)
            x_dequant = x_q * scale
            return x_dequant
        else:
            return x_clipped
