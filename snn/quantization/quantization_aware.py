"""
Quantization-Aware Training for SNNs
Supports variable bit-width quantization with hardware deployment in mind
"""

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
        ema_decay: float = 0.9,
        enable_logging: bool = False,
        layer_name: str = "unknown",
        logger: Optional[Any] = None,  # TensorBoard logger instance
        quantize: bool = True  # Whether to actually quantize or just log
    ):
        super().__init__()
        
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.ema_decay = ema_decay
        self.enable_logging = enable_logging
        self.layer_name = layer_name
        self.logger = logger
        self.quantize = quantize
        self.forward_count = 0
        
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
        
        # Track peak values to prevent collapse (critical for SNNs with sparse activations)
        self.register_buffer('peak_min', torch.tensor(0.0))
        self.register_buffer('peak_max', torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization with EMA statistics tracking"""
        
        if self.training:
            # Update running statistics using EMA
            # CRITICAL FOR SNNs: Only update if there are non-zero activations
            # to prevent collapse when most activations are zero (no spikes)
            with torch.no_grad():
                # Check if we have meaningful activations
                abs_max = x.abs().max()
                
                if abs_max > 1e-3:  # Only update if there are actual spikes/activations
                    min_val = x.min()
                    max_val = x.max()
                    
                    if self.num_batches_tracked == 0:
                        # Initialize both running and peak statistics
                        self.running_min = min_val
                        self.running_max = max_val
                        self.peak_min = min_val
                        self.peak_max = max_val
                    else:
                        # Update EMA statistics (faster decay for SNNs)
                        self.running_min = self.ema_decay * self.running_min + (1 - self.ema_decay) * min_val
                        self.running_max = self.ema_decay * self.running_max + (1 - self.ema_decay) * max_val
                        
                        # Update peak statistics (never decay, only expand)
                        self.peak_min = torch.min(self.peak_min, min_val)
                        self.peak_max = torch.max(self.peak_max, max_val)
                    
                    self.num_batches_tracked += 1
        
        # Use running statistics with peak floor to prevent collapse
        # Floor prevents EMA from decaying too far when sparse activations occur
        effective_min = torch.max(self.running_min, self.peak_min * 0.5)  # Allow EMA to be at most 50% of peak
        effective_max = torch.max(self.running_max, self.peak_max * 0.5)
        
        # Compute scale and zero-point
        if self.symmetric:
            max_abs = torch.max(torch.abs(effective_min), torch.abs(effective_max))
            scale = max_abs / (2 ** (self.bit_width - 1) - 1)
            zero_point = 0.0
        else:
            scale = (effective_max - effective_min) / (2 ** self.bit_width - 1)
            zero_point = effective_min
        
        # Enforce minimum scale to prevent collapse (increased from 1e-4 to 0.01)
        min_scale = 0.01  # More aggressive floor for SNN stability
        scale = torch.clamp(scale, min=min_scale)
        
        # Fake quantization: quantize then dequantize (only if quantize flag is True)
        if self.quantize and scale > 1e-8:  # Avoid division by zero
            x_q = (x - zero_point) / scale
            x_q = torch.clamp(x_q, self.qmin, self.qmax)
            x_q = StraightThroughEstimator.apply(x_q)
            x_dequant = x_q * scale + zero_point
        else:
            # No quantization, pass through as-is
            x_dequant = x
            
        # Log statistics to TensorBoard if enabled (regardless of quantization mode)
        if self.enable_logging and self.training and self.logger is not None and self.forward_count % 100 == 0:
            with torch.no_grad():
                # Log activation statistics
                self.logger.log_scalars(f'params/{self.layer_name}/activations/input', {
                    'min': x.min().item(),
                    'max': x.max().item(),
                    'mean': x.mean().item(),
                    'std': x.std().item()
                }, self.forward_count)
                
                self.logger.log_scalars(f'params/{self.layer_name}/activations/output', {
                    'min': x_dequant.min().item(),
                    'max': x_dequant.max().item(),
                    'mean': x_dequant.mean().item(),
                    'std': x_dequant.std().item()
                }, self.forward_count)
                
                # Log quantization parameters (even if not quantizing, shows what they would be)
                self.logger.log_scalars(f'params/{self.layer_name}/activations/params', {
                    'scale': scale.item(),
                    'running_max': self.running_max.item(),
                    'peak_max': self.peak_max.item(),
                }, self.forward_count)
                
                # Log histogram of activations
                self.logger.log_histogram(f'params/{self.layer_name}/activations/input_hist', x, self.forward_count)
                self.logger.log_histogram(f'params/{self.layer_name}/activations/output_hist', x_dequant, self.forward_count)
        
        self.forward_count += 1
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


class QuantizedWeight(torch.autograd.Function):
    """
    Multi-bit weight quantization with per-channel scaling
    
    Uses symmetric quantization with per-channel (per-filter) scaling
    for better accuracy compared to per-tensor scaling.
    """
    @staticmethod
    def forward(ctx, weight, bit_width):
        # Per-channel (per-output-filter) quantization
        # Shape: [out_ch, in_ch, kH, kW] -> scale per out_ch
        
        # Calculate per-channel max absolute value
        # Need to flatten spatial and input channel dimensions for max
        w_reshaped = weight.abs().view(weight.size(0), -1)  # [out_ch, in_ch*kH*kW]
        max_abs = w_reshaped.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)  # [out_ch, 1, 1, 1]
        
        # Symmetric quantization range
        qmax = 2 ** (bit_width - 1) - 1
        
        # Calculate scale
        scale = max_abs / qmax
        scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
        
        # Quantize
        w_q = weight / scale
        w_q = torch.clamp(w_q, -qmax, qmax)
        w_q = w_q.round()
        
        # Dequantize
        w_dequant = w_q * scale
        
        return w_dequant
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None


class QuantizedConv2d(nn.Module):
    """
    Quantized Conv2d that quantizes both weights and activations
    
    This is the proper way to do QAT for hardware deployment:
    - Weights are quantized during forward pass (stored as FP32, quantized on-the-fly)
    - Activations are quantized after convolution
    - Both use straight-through estimators for gradients
    
    Args:
        Same as nn.Conv2d, plus:
        weight_bit_width: Bit-width for weight quantization
        act_bit_width: Bit-width for activation quantization
        quantize_weights: Enable weight quantization
        quantize_activations: Enable activation quantization
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        quantize_weights: bool = True,
        quantize_activations: bool = True,
        enable_logging: bool = False,
        layer_name: str = "conv",
        logger: Optional[Any] = None  # TensorBoard logger instance
    ):
        super().__init__()
        
        # Separate bit-widths for weights and activations
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.enable_logging = enable_logging
        self.layer_name = layer_name
        self.logger = logger
        self.forward_count = 0
        
        # Standard conv layer (weights stored in FP32)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        
        # Weight scale tracking (per-channel, shape: [out_channels, 1, 1, 1])
        # Initialize after conv layer so we know out_channels
        if self.quantize_weights:
            self.register_buffer('weight_scale', torch.ones(out_channels, 1, 1, 1))
            self.register_buffer('weight_scale_initialized', torch.tensor(False))
        
        # Activation quantization layer (always present for logging)
        # When quantization is disabled, it still logs but doesn't quantize
        self.act_quant = QuantizationAwareLayer(
            bit_width=self.act_bit_width,
            enable_logging=enable_logging,
            layer_name=layer_name,  # Use base layer name, not layer_name_act
            logger=logger,
            quantize=quantize_activations  # Pass quantization flag
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights and activations
        """
        # Quantize weights if enabled (using weight_bit_width)
        if self.quantize_weights and self.weight_bit_width > 1:
            # Multi-bit weight quantization with persistent scale
            # Initialize scale on first forward pass
            if not self.weight_scale_initialized:
                with torch.no_grad():
                    # Calculate initial per-channel max absolute value
                    w_reshaped = self.conv.weight.abs().view(self.conv.weight.size(0), -1)
                    max_abs = w_reshaped.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
                    
                    # Set scale based on initial weights
                    qmax = 2 ** (self.weight_bit_width - 1) - 1
                    self.weight_scale.copy_(max_abs / qmax)
                    self.weight_scale.clamp_(min=1e-4)  # Minimum scale to prevent collapse
                    self.weight_scale_initialized.fill_(True)
            
            # Quantize using FIXED scale (doesn't change with weight values)
            qmax = 2 ** (self.weight_bit_width - 1) - 1
            w_q = self.conv.weight / self.weight_scale
            w_q = torch.clamp(w_q, -qmax, qmax)
            w_q = StraightThroughEstimator.apply(w_q.round())
            weight = w_q * self.weight_scale
            
        elif self.quantize_weights and self.weight_bit_width == 1:
            # Binary weight quantization
            weight = BinaryWeight.apply(self.conv.weight)
        else:
            # No weight quantization
            weight = self.conv.weight
        
        # Log quantized/processed weight statistics to TensorBoard (regardless of quantization mode)
        if self.enable_logging and self.training and self.logger is not None and self.forward_count % 100 == 0:
            with torch.no_grad():
                self.logger.log_scalars(f'params/{self.layer_name}/weights', {
                    'min': weight.min().item(),
                    'max': weight.max().item(),
                    'mean': weight.mean().item(),
                    'std': weight.std().item()
                }, self.forward_count)
                self.logger.log_histogram(f'params/{self.layer_name}/weights_hist', weight, self.forward_count)
        
        # Perform convolution with quantized weights
        out = F.conv2d(
            x, weight, self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )
        
        # Quantize activations (or pass through identity if disabled)
        out = self.act_quant(out)
        
        self.forward_count += 1
        return out
