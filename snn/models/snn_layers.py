"""
Spiking Neural Network Layer Implementations
Includes LIF neurons and spiking convolution layers with quantization support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .quant_layers import *

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


def quantize_membrane(mem, bit_width=8, mem_range=2.0):
    """
    Quantize membrane potential for hardware implementation
    
    Args:
        mem: Membrane potential tensor
        bit_width: Number of bits for quantization
        mem_range: Expected range of membrane values (±mem_range)
    
    Returns:
        Quantized membrane potential
    """
    # Symmetric quantization around zero
    qmax = 2 ** (bit_width - 1) - 1
    scale = mem_range / qmax
    
    # Quantize
    mem_q = mem / scale
    mem_q = torch.clamp(mem_q, -qmax, qmax)
    mem_q = mem_q.round()
    
    # Dequantize (STE for gradients)
    mem_dequant = mem_q * scale
    
    return mem_dequant


# -------------------------
# LIF neuron update
# -------------------------
def lif_update(mem, inp, decay=0.5, threshold=1.0, alpha=10.0,
               quantize_mem=False, mem_bit_width=8):
    """
    LIF neuron membrane update
    
    Args:
        mem: membrane state
        inp: input current (conv output)
        decay: membrane decay factor (0-1, smaller = faster decay)
        threshold: spike threshold
        alpha: surrogate gradient slope
        quantize_mem: Enable membrane potential quantization
        mem_bit_width: Bit width for membrane quantization
    """
    if mem is None:
        mem = torch.zeros_like(inp)

    # Use decay factor directly
    decay_factor = torch.tensor(decay, device=inp.device, dtype=inp.dtype)
    mem = mem * decay_factor + inp

    spk = spike_fn(mem - threshold, alpha=alpha)
    # reset by subtracting threshold on spike (soft reset)
    mem = mem - spk * mem
    
    # Quantize membrane again after reset if enabled
    if quantize_mem:
        mem = quantize_membrane(mem, bit_width=mem_bit_width, mem_range=threshold * 2.0)
    
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
        decay=0.5,
        threshold=1.0,
        alpha=10.0,
        quantize_weights=False,
        quantize_activations=False,
        quantize_mem=False,
        weight_bit_width=8,
        act_bit_width=8,
        mem_bit_width=16,
        enable_logging=False,
        layer_name="spiking_conv",
        logger=None  # TensorBoard logger instance
    ):
        super().__init__()
        
        # Quantization support - use QuantizedConv2d for full quantization
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.quantize_mem = quantize_mem
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width
        self.mem_bit_width = mem_bit_width
        
        # Always use QuantizedConv2d for consistent structure
        # When quantize=False, it acts as a standard conv
        self.conv = QuantizedConv2d(
            in_ch, out_ch, kernel_size=k, stride=s, padding=p, 
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            enable_logging=enable_logging,
            layer_name=layer_name,
            logger=logger
        )
        
        self.decay = decay
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, x, mem):
        # Convolution (with weight and activation quantization if enabled)
        x = self.conv(x)
        
        # LIF neuron dynamics (hardware-friendly in hardware_mode, with membrane quantization)
        spk, mem = lif_update(mem, x, decay=self.decay, threshold=self.threshold, 
                             alpha=self.alpha,
                             quantize_mem=self.quantize_mem, mem_bit_width=self.mem_bit_width)
        return spk, mem

