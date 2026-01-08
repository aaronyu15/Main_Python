"""
Binary Spiking Neural Network Layers
Extreme quantization (1-bit weights and activations) for FPGA deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .quantization_aware import BinaryWeight, BinaryActivation


class BinaryLIF(nn.Module):
    """
    Binary LIF Neuron
    Uses binary threshold and simplified dynamics for hardware efficiency
    """
    def __init__(
        self,
        tau: float = 2.0,
        threshold: float = 1.0,
        hard_reset: bool = True
    ):
        super().__init__()
        self.register_buffer('beta', torch.tensor(1.0 - 1.0/tau))
        self.threshold = threshold
        self.hard_reset = hard_reset
    
    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Binary LIF forward pass"""
        if mem is None:
            mem = torch.zeros_like(x)
        
        # Binary leak (shift operation in hardware)
        mem = self.beta * mem + x
        
        # Binary spike generation
        spikes = (mem >= self.threshold).float()
        
        # Hard reset (efficient in hardware)
        if self.hard_reset:
            mem = mem * (1.0 - spikes)
        else:
            mem = mem - spikes * self.threshold
        
        return spikes, mem


class BinarySpikeConv2d(nn.Module):
    """
    Binary Spiking Convolutional Layer
    1-bit weights and activations for extreme FPGA efficiency
    
    Operations are reduced to XNOR and popcount, which are very efficient in hardware
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        tau: float = 2.0,
        threshold: float = 1.0
    ):
        super().__init__()
        
        # Standard conv layer (weights will be binarized)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # No bias for binary networks
        )
        
        # Batch normalization (helps with binary training)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        
        # Binary LIF neuron
        self.lif = BinaryLIF(tau=tau, threshold=threshold)
        
        # Learnable scaling factors
        self.register_parameter('alpha', nn.Parameter(torch.ones(out_channels, 1, 1)))
        
    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with binary weights
        
        In hardware:
        - Binary weights stored as 1 bit each
        - Convolution becomes XNOR + popcount
        - Very fast and energy efficient
        """
        # Binarize weights during forward pass
        binary_weight = BinaryWeight.apply(self.conv.weight)
        
        # Binary convolution
        out = F.conv2d(
            x, binary_weight,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding
        )
        
        # Scaling
        out = out * self.alpha
        
        # Batch norm
        out = self.bn(out)
        
        # Binary LIF
        spikes, mem = self.lif(out, mem)
        
        return spikes, mem


class BinarySpikingResBlock(nn.Module):
    """
    Binary Spiking Residual Block
    For deeper binary SNNs
    """
    def __init__(
        self,
        channels: int,
        tau: float = 2.0,
        threshold: float = 1.0
    ):
        super().__init__()
        
        self.conv1 = BinarySpikeConv2d(
            channels, channels,
            kernel_size=3, padding=1,
            tau=tau, threshold=threshold
        )
        self.conv2 = BinarySpikeConv2d(
            channels, channels,
            kernel_size=3, padding=1,
            tau=tau, threshold=threshold
        )
    
    def forward(self, x: torch.Tensor, mem1: Optional[torch.Tensor] = None,
                mem2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with binary residual connection"""
        identity = x
        
        out, mem1 = self.conv1(x, mem1)
        out, mem2 = self.conv2(out, mem2)
        
        # Binary residual (XOR in hardware)
        out = out + identity
        
        return out, mem1, mem2


class BinaryEncoder(nn.Module):
    """
    Binary encoder for input preprocessing
    Converts analog inputs to binary spike trains
    """
    def __init__(self, method: str = 'rate'):
        """
        Args:
            method: 'rate' for rate coding, 'temporal' for temporal coding
        """
        super().__init__()
        self.method = method
    
    def forward(self, x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        Encode input to binary spikes
        
        Args:
            x: Input tensor (normalized to [0, 1])
            num_steps: Number of timesteps
            
        Returns:
            Binary spike train [T, B, C, H, W]
        """
        if self.method == 'rate':
            # Rate coding: spike probability proportional to input
            spikes = []
            for _ in range(num_steps):
                spike = (torch.rand_like(x) < x).float()
                spikes.append(spike)
            return torch.stack(spikes, dim=0)
        
        elif self.method == 'temporal':
            # Temporal coding: spike timing encodes value
            spikes = []
            threshold = torch.linspace(1.0, 0.0, num_steps, device=x.device)
            for t in range(num_steps):
                spike = (x >= threshold[t]).float()
                spikes.append(spike)
            return torch.stack(spikes, dim=0)
        
        else:
            raise ValueError(f"Unknown encoding method: {self.method}")


class XNOR_Net(nn.Module):
    """
    XNOR-Net style binary neural network
    Optimized for hardware implementation
    
    Reference: "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks"
    """
    def __init__(self):
        super().__init__()
        # Implementation note: In hardware, this uses XNOR gates and bit-counting
        # instead of multiplications
        pass
    
    @staticmethod
    def binary_conv2d(input: torch.Tensor, weight: torch.Tensor, 
                      stride: int = 1, padding: int = 0) -> torch.Tensor:
        """
        Binary convolution using XNOR and popcount
        
        In hardware:
        1. XNOR between binary input and binary weight
        2. Popcount (count 1s) to get convolution result
        3. Scale and shift
        
        This is ~58x faster and ~32x more energy efficient than float conv
        """
        # Binarize
        binary_input = torch.sign(input)
        binary_weight = torch.sign(weight)
        
        # Calculate scaling factors
        K = weight.shape[1] * weight.shape[2] * weight.shape[3]  # kernel size
        alpha = weight.abs().mean(dim=[1, 2, 3], keepdim=True)
        beta = input.abs().mean(dim=[1, 2, 3], keepdim=True)
        
        # Binary convolution (XNOR + popcount)
        output = F.conv2d(binary_input, binary_weight, stride=stride, padding=padding)
        
        # Scale
        output = output * alpha * beta
        
        return output
