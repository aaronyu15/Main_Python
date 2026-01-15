"""
Spiking Neural Network Layer Implementations
Includes LIF neurons and spiking convolution layers with quantization support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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
        groups=1,
        quantize=False,
        bit_width=8
    ):
        super().__init__()
        
        # Quantization support - use QuantizedConv2d for full quantization
        self.quantize = quantize
        self.bit_width = bit_width
        
        if quantize:
            # Use QuantizedConv2d that quantizes BOTH weights and activations
            from ..quantization import QuantizedConv2d
            self.conv = QuantizedConv2d(
                in_ch, out_ch, kernel_size=k, stride=s, padding=p, 
                groups=groups, bias=not use_bn,
                bit_width=bit_width,
                quantize_weights=True,
                quantize_activations=True
            )
        else:
            # Standard full-precision convolution
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=not use_bn)
        
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else None
        self.tau = tau
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, x, mem):
        # Convolution (with weight and activation quantization if enabled)
        x = self.conv(x)
        
        # Batch normalization
        if self.bn is not None:
            x = self.bn(x)
        
        # LIF neuron dynamics
        spk, mem = lif_update(mem, x, tau=self.tau, threshold=self.threshold, alpha=self.alpha)
        return spk, mem

