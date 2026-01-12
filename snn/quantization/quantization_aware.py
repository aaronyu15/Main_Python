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
    Fake Quantization layer for Quantization-Aware Training
    
    Uses exponential moving average (EMA) to track activation statistics
    during training, similar to PyTorch's FakeQuantize.
    
    Args:
        bit_width: Number of bits for quantization (1 for binary)
        symmetric: Use symmetric quantization around zero
        ema_decay: Decay factor for exponential moving average (0.9-0.999)
    """
    def __init__(
        self,
        bit_width: int = 8,
        symmetric: bool = True,
        ema_decay: float = 0.99
    ):
        super().__init__()
        
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.ema_decay = ema_decay
        
        # Quantization levels
        if symmetric:
            self.qmin = -(2 ** (bit_width - 1))
            self.qmax = 2 ** (bit_width - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bit_width - 1
        
        # Running statistics for scale/zero-point (like BatchNorm)
        self.register_buffer('running_min', torch.tensor(0.0))
        self.register_buffer('running_max', torch.tensor(1.0))
        self.register_buffer('num_batches_tracked', torch.tensor(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization with EMA statistics tracking"""
        
        if self.training:
            # Update running statistics using EMA
            with torch.no_grad():
                min_val = x.min()
                max_val = x.max()
                
                if self.num_batches_tracked == 0:
                    self.running_min = min_val
                    self.running_max = max_val
                else:
                    self.running_min = self.ema_decay * self.running_min + (1 - self.ema_decay) * min_val
                    self.running_max = self.ema_decay * self.running_max + (1 - self.ema_decay) * max_val
                
                self.num_batches_tracked += 1
        
        # Use running statistics for quantization
        if self.symmetric:
            max_abs = torch.max(torch.abs(self.running_min), torch.abs(self.running_max))
            scale = max_abs / (2 ** (self.bit_width - 1) - 1)
            zero_point = 0.0
        else:
            scale = (self.running_max - self.running_min) / (2 ** self.bit_width - 1)
            zero_point = self.running_min
        
        # Fake quantization: quantize then dequantize
        if scale > 1e-8:  # Avoid division by zero
            x_q = (x - zero_point) / scale
            x_q = torch.clamp(x_q, self.qmin, self.qmax)
            x_q = StraightThroughEstimator.apply(x_q)
            x_dequant = x_q * scale + zero_point
            return x_dequant
        else:
            return x


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
