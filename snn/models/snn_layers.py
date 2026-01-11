"""
Spiking Neural Network Layer Implementations
Includes LIF neurons and spiking convolution layers with quantization support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron
    
    Args:
        tau: Membrane time constant
        threshold: Firing threshold
        reset_mode: 'subtract' or 'zero' - how to reset membrane potential
        surrogate_gradient: Use surrogate gradient for backpropagation
    """
    def __init__(
        self, 
        tau: float = 2.0,
        threshold: float = 1.0,
        reset_mode: str = 'subtract',
        surrogate_gradient: bool = True
    ):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.reset_mode = reset_mode
        self.surrogate_gradient = surrogate_gradient
        
        # Learnable parameters option
        self.register_buffer('beta', torch.tensor(1.0 - 1.0/tau))
        
    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LIF neuron
        
        Args:
            x: Input current [B, C, H, W]
            mem: Previous membrane potential [B, C, H, W]
            
        Returns:
            spikes: Output spikes [B, C, H, W]
            mem: Updated membrane potential [B, C, H, W]
        """
        if mem is None:
            mem = torch.zeros_like(x)
            
        # Leaky integration
        mem = self.beta * mem + x
        
        # Spike generation with surrogate gradient
        if self.surrogate_gradient:
            spikes = SpikeFunctionBoxcar.apply(mem - self.threshold, self.threshold)
        else:
            spikes = (mem >= self.threshold).float()
        
        # Reset mechanism
        if self.reset_mode == 'subtract':
            mem = mem - spikes * self.threshold
        elif self.reset_mode == 'zero':
            mem = mem * (1 - spikes)
        
        return spikes, mem


class SpikeFunctionBoxcar(torch.autograd.Function):
    """
    Surrogate gradient for spike function using boxcar (rectangular) function
    Forward: Heaviside step function
    Backward: Boxcar function (rectangular window)
    """
    @staticmethod
    def forward(ctx, x, width=1.0):
        ctx.save_for_backward(x)
        ctx.width = width
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Boxcar surrogate gradient
        grad_input = grad_output.clone()
        grad_input[torch.abs(x) > ctx.width/2] = 0
        return grad_input / ctx.width, None


class SpikingConv2d(nn.Module):
    """
    Spiking Convolutional Layer with LIF neuron
    Combines Conv2d + BatchNorm + LIF
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        bias: Use bias in convolution
        tau: LIF neuron time constant
        threshold: LIF firing threshold
        quantize: Enable quantization-aware training
        bit_width: Bit width for quantization (if enabled)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        tau: float = 2.0,
        threshold: float = 1.0,
        quantize: bool = False,
        bit_width: int = 8
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = LIFNeuron(tau=tau, threshold=threshold)
        
        # Quantization parameters
        self.quantize = quantize
        self.bit_width = bit_width
        if quantize:
            from ..quantization import QuantizationAwareLayer
            self.quant_layer = QuantizationAwareLayer(bit_width=bit_width)
        
    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through spiking conv layer
        
        Args:
            x: Input tensor [B, C, H, W] or spikes from previous layer
            mem: Membrane potential from previous timestep
            
        Returns:
            spikes: Output spikes
            mem: Updated membrane potential
        """
        # Convolution
        out = self.conv(x)
        
        # Apply quantization if enabled
        if self.quantize:
            out = self.quant_layer(out)
        
        # Batch normalization
        out = self.bn(out)
        
        # LIF neuron dynamics
        spikes, mem = self.lif(out, mem)
        
        return spikes, mem


class SpikingConvTranspose2d(nn.Module):
    """
    Spiking Transposed Convolutional Layer (for upsampling)
    Combines ConvTranspose2d + BatchNorm + LIF
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        tau: float = 2.0,
        threshold: float = 1.0,
        quantize: bool = False,
        bit_width: int = 8
    ):
        super().__init__()
        
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = LIFNeuron(tau=tau, threshold=threshold)
        
        # Quantization parameters
        self.quantize = quantize
        self.bit_width = bit_width
        if quantize:
            from ..quantization import QuantizationAwareLayer
            self.quant_layer = QuantizationAwareLayer(bit_width=bit_width)
        
    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through spiking transposed conv layer"""
        out = self.conv_transpose(x)
        
        if self.quantize:
            out = self.quant_layer(out)
        
        out = self.bn(out)
        spikes, mem = self.lif(out, mem)
        
        return spikes, mem


class SpikingResidualBlock(nn.Module):
    """
    Spiking Residual Block for deeper networks
    """
    def __init__(
        self, 
        channels: int,
        tau: float = 2.0,
        threshold: float = 1.0,
        quantize: bool = False,
        bit_width: int = 8
    ):
        super().__init__()
        
        self.conv1 = SpikingConv2d(
            channels, channels, 
            kernel_size=3, padding=1,
            tau=tau, threshold=threshold,
            quantize=quantize, bit_width=bit_width
        )
        self.conv2 = SpikingConv2d(
            channels, channels,
            kernel_size=3, padding=1,
            tau=tau, threshold=threshold,
            quantize=quantize, bit_width=bit_width
        )
        
    def forward(self, x: torch.Tensor, mem1: Optional[torch.Tensor] = None, 
                mem2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with residual connection"""
        identity = x
        
        out, mem1 = self.conv1(x, mem1)
        out, mem2 = self.conv2(out, mem2)
        
        # Residual connection (spike addition)
        out = out + identity
        
        return out, mem1, mem2

# -------------------------
# Surrogate spike function
# -------------------------
class SurrogateSpike(torch.autograd.Function):
    """
    Hard threshold in forward; smooth surrogate gradient in backward.
    """
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        # derivative of sigmoid(alpha*x): alpha * s * (1-s)
        s = torch.sigmoid(alpha * x)
        grad = alpha * s * (1 - s)
        return grad_out * grad, None


def spike_fn(x, alpha=10.0):
    return SurrogateSpike.apply(x, alpha)


# -------------------------
# LIF neuron update
# -------------------------
def lif_update(mem, inp, tau=2.0, threshold=1.0, alpha=10.0):
    """
    mem: membrane state
    inp: input current (conv output)
    """
    if mem is None:
        mem = torch.zeros_like(inp)

    # simple discrete-time decay (hardware-friendly)
    decay = torch.exp(torch.tensor(-1.0 / tau, device=inp.device, dtype=inp.dtype))
    mem = mem * decay + inp

    spk = spike_fn(mem - threshold, alpha=alpha)
    # reset by subtracting threshold on spike (soft reset)
    mem = mem - spk * threshold
    return spk, mem


# -------------------------
# Spiking Conv Block
# -------------------------
class SpikingConvBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        k=3,
        s=1,
        p=1,
        tau=2.0,
        threshold=1.0,
        alpha=10.0,
        use_bn=False,
        groups=1
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else None
        self.tau = tau
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, x, mem):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        spk, mem = lif_update(mem, x, tau=self.tau, threshold=self.threshold, alpha=self.alpha)
        return spk, mem

