"""
Spiking FlowNet for Optical Flow Estimation
Event-based/spike-based architecture for optical flow prediction
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .snn_layers import SpikingConv2d, SpikingConvTranspose2d, SpikingResidualBlock, SpikingConvBlock
import torch.nn.functional as F

class SpikingFlowNetLite(nn.Module):
    """
    Lightweight Spiking FlowNet for FPGA deployment
    Reduced parameter count and depth for hardware efficiency
    """
    def __init__(
        self,
        in_channels: int = 2,
        num_timesteps: int = 10,
        tau: float = 2.0,
        threshold: float = 1.0,
        quantize: bool = True,
        bit_width: int = 4,
        binarize: bool = False
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.bit_width = 1 if binarize else bit_width
        
        # Simplified encoder
        self.enc1 = SpikingConv2d(in_channels, 32, 5, 2, 2, tau=tau, threshold=threshold, 
                                  quantize=quantize, bit_width=self.bit_width)
        self.enc2 = SpikingConv2d(32, 64, 3, 2, 1, tau=tau, threshold=threshold,
                                  quantize=quantize, bit_width=self.bit_width)
        self.enc3 = SpikingConv2d(64, 128, 3, 2, 1, tau=tau, threshold=threshold,
                                  quantize=quantize, bit_width=self.bit_width)
        
        # Simplified decoder
        self.dec3 = SpikingConvTranspose2d(128, 64, 4, 2, 1, tau=tau, threshold=threshold,
                                           quantize=quantize, bit_width=self.bit_width)
        self.dec2 = SpikingConvTranspose2d(128, 32, 4, 2, 1, tau=tau, threshold=threshold,
                                           quantize=quantize, bit_width=self.bit_width)
        
        # Flow prediction
        self.flow_head = nn.Conv2d(32, 2, 3, 1, 1)
        
        # Scaling factor to compensate for small spike accumulation values
        # With binary spikes over num_timesteps, accumulated values are small
        # This scales the output to match expected flow magnitudes
        self.flow_scale = 20.0
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for lightweight model"""
        batch_size, _, height, width = x.shape
        
        # Initialize membrane potentials
        mem_e1, mem_e2, mem_e3 = None, None, None
        mem_d3, mem_d2 = None, None
        
        # Accumulators
        acc_e1 = torch.zeros(batch_size, 32, height//2, width//2, device=x.device)
        acc_e2 = torch.zeros(batch_size, 64, height//4, width//4, device=x.device)
        acc_e3 = torch.zeros(batch_size, 128, height//8, width//8, device=x.device)
        
        # Run through timesteps
        for t in range(self.num_timesteps):
            s1, mem_e1 = self.enc1(x, mem_e1)
            s2, mem_e2 = self.enc2(s1, mem_e2)
            s3, mem_e3 = self.enc3(s2, mem_e3)
            
            acc_e1 += s1
            acc_e2 += s2
            acc_e3 += s3
        
        # Decode
        d3, mem_d3 = self.dec3(acc_e3, mem_d3)
        d3_cat = torch.cat([d3, acc_e2], dim=1)
        
        d2, mem_d2 = self.dec2(d3_cat, mem_d2)
        
        # Flow prediction with scaling
        flow = self.flow_head(d2)
        flow = flow * self.flow_scale  # Scale up to match expected flow magnitudes
        flow = nn.functional.interpolate(flow, size=(height, width), mode='bilinear', align_corners=False)
        
        return {'flow': flow}



# -------------------------
# SNN-friendly Flow Net
# -------------------------
class EventSNNFlowNetLite(nn.Module):
    """
    Input:  x [N, T, 2, H, W]  (T time bins, 2 polarities)
    Output: flow [N, 2, H, W]
    
    Supports quantization-aware training for hardware deployment
    """
    def __init__(
        self,
        base_ch=32,
        tau=2.0,
        threshold=1.0,
        alpha=10.0,
        use_bn=False,
        quantize=False,
        bit_width=8,
        binarize=False
    ):
        super().__init__()
        
        # Quantization settings
        self.quantize = quantize
        self.bit_width = 1 if binarize else bit_width

        # Encoder (downsample by 2 each stage)
        self.e1 = SpikingConvBlock(2, base_ch,     k=5, s=2, p=2, tau=tau, threshold=threshold, alpha=alpha, use_bn=use_bn, quantize=quantize, bit_width=self.bit_width)
        self.e2 = SpikingConvBlock(base_ch, base_ch*2, k=3, s=2, p=1, tau=tau, threshold=threshold, alpha=alpha, use_bn=use_bn, quantize=quantize, bit_width=self.bit_width)
        self.e3 = SpikingConvBlock(base_ch*2, base_ch*4, k=3, s=2, p=1, tau=tau, threshold=threshold, alpha=alpha, use_bn=use_bn, quantize=quantize, bit_width=self.bit_width)

        # Decoder (upsample + spiking conv)
        # Using conv after upsample tends to be friendlier than ConvTranspose2d on hardware.
        self.d3 = SpikingConvBlock(base_ch*4, base_ch*2, k=3, s=1, p=1, tau=tau, threshold=threshold, alpha=alpha, use_bn=use_bn, quantize=quantize, bit_width=self.bit_width)
        self.d2 = SpikingConvBlock(base_ch*2 + base_ch*2, base_ch, k=3, s=1, p=1, tau=tau, threshold=threshold, alpha=alpha, use_bn=use_bn, quantize=quantize, bit_width=self.bit_width)
        self.d1 = SpikingConvBlock(base_ch + base_ch, base_ch, k=3, s=1, p=1, tau=tau, threshold=threshold, alpha=alpha, use_bn=use_bn, quantize=quantize, bit_width=self.bit_width)

        # Non-spiking regression head (keep it simple & stable)
        self.flow_head = nn.Conv2d(base_ch, 2, kernel_size=3, padding=1)

        # optional scale (you can tune or learn this)
        self.flow_scale = 20.0

    def forward(self, x):
        """
        x: [N,T,2,H,W]  (recommended)
           OR [N,2*T,H,W] (we'll reshape to time bins if you pass this)
        """
        if x.dim() == 4:
            # assume channel-packed voxel grid: [N, 2*T, H, W]
            N, CT, H, W = x.shape
            assert CT % 2 == 0, "Expected channels = 2*T"
            T = CT // 2
            x = x.view(N, T, 2, H, W)
        else:
            N, T, C, H, W = x.shape
            assert C == 2, "Expected 2 polarity channels"

        # Mem states
        mem_e1 = mem_e2 = mem_e3 = None
        mem_d3 = mem_d2 = mem_d1 = None

        # Spike accumulators for skip connections (rate coding over T bins)
        acc_e1 = torch.zeros((N, self.e1.conv.out_channels, H//2, W//2), device=x.device)
        acc_e2 = torch.zeros((N, self.e2.conv.out_channels, H//4, W//4), device=x.device)
        acc_e3 = torch.zeros((N, self.e3.conv.out_channels, H//8, W//8), device=x.device)

        # Process real time bins
        for t in range(T):
            xt = x[:, t]  # [N,2,H,W]

            s1, mem_e1 = self.e1(xt, mem_e1)   # [N,base,H/2,W/2]
            s2, mem_e2 = self.e2(s1, mem_e2)   # [N,2base,H/4,W/4]
            s3, mem_e3 = self.e3(s2, mem_e3)   # [N,4base,H/8,W/8]

            acc_e1 += s1
            acc_e2 += s2
            acc_e3 += s3

        # Normalize by T to make magnitude less dependent on bin count
        acc_e1 = acc_e1 / T
        acc_e2 = acc_e2 / T
        acc_e3 = acc_e3 / T

        # Decode
        d3 = F.interpolate(acc_e3, scale_factor=2, mode="nearest")   # -> H/4
        d3, mem_d3 = self.d3(d3, mem_d3)

        d3_cat = torch.cat([d3, acc_e2], dim=1)                      # skip
        d2 = F.interpolate(d3_cat, scale_factor=2, mode="nearest")   # -> H/2
        d2, mem_d2 = self.d2(d2, mem_d2)

        d2_cat = torch.cat([d2, acc_e1], dim=1)                      # skip
        d1 = F.interpolate(d2_cat, scale_factor=2, mode="nearest")   # -> H
        d1, mem_d1 = self.d1(d1, mem_d1)

        flow = self.flow_head(d1) * self.flow_scale
        return {"flow": flow}

