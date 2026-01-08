"""
Spiking FlowNet for Optical Flow Estimation
Event-based/spike-based architecture for optical flow prediction
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .snn_layers import SpikingConv2d, SpikingConvTranspose2d, SpikingResidualBlock


class SpikingFlowNet(nn.Module):
    """
    Spiking Neural Network for Optical Flow Estimation
    
    Architecture inspired by FlowNet but adapted for spiking neurons.
    Uses encoder-decoder structure with skip connections.
    
    Args:
        in_channels: Number of input channels (e.g., 2 for stacked events)
        num_timesteps: Number of timesteps to simulate
        tau: LIF neuron time constant
        threshold: LIF firing threshold
        quantize: Enable quantization-aware training
        bit_width: Bit width for quantization
        binarize: Use binary weights and activations (extreme quantization)
    """
    def __init__(
        self,
        in_channels: int = 2,
        num_timesteps: int = 10,
        tau: float = 2.0,
        threshold: float = 1.0,
        quantize: bool = False,
        bit_width: int = 8,
        binarize: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_timesteps = num_timesteps
        self.quantize = quantize
        self.bit_width = 1 if binarize else bit_width
        self.binarize = binarize
        
        # Encoder (contracting path)
        self.encoder1 = SpikingConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                                      tau=tau, threshold=threshold, quantize=quantize, bit_width=self.bit_width)
        self.encoder2 = SpikingConv2d(64, 128, kernel_size=5, stride=2, padding=2,
                                      tau=tau, threshold=threshold, quantize=quantize, bit_width=self.bit_width)
        self.encoder3 = SpikingConv2d(128, 256, kernel_size=5, stride=2, padding=2,
                                      tau=tau, threshold=threshold, quantize=quantize, bit_width=self.bit_width)
        self.encoder4 = SpikingConv2d(256, 512, kernel_size=3, stride=2, padding=1,
                                      tau=tau, threshold=threshold, quantize=quantize, bit_width=self.bit_width)
        self.encoder5 = SpikingConv2d(512, 512, kernel_size=3, stride=2, padding=1,
                                      tau=tau, threshold=threshold, quantize=quantize, bit_width=self.bit_width)
        
        # Decoder (expanding path)
        self.decoder5 = SpikingConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1,
                                               tau=tau, threshold=threshold, quantize=quantize, bit_width=self.bit_width)
        self.decoder4 = SpikingConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1,  # 512 + 512
                                               tau=tau, threshold=threshold, quantize=quantize, bit_width=self.bit_width)
        self.decoder3 = SpikingConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1,   # 256 + 256
                                               tau=tau, threshold=threshold, quantize=quantize, bit_width=self.bit_width)
        self.decoder2 = SpikingConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1,    # 128 + 128
                                               tau=tau, threshold=threshold, quantize=quantize, bit_width=self.bit_width)
        
        # Flow prediction heads at multiple scales
        self.flow_head5 = nn.Conv2d(512, 2, kernel_size=3, padding=1)
        self.flow_head4 = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        self.flow_head3 = nn.Conv2d(128, 2, kernel_size=3, padding=1)
        self.flow_head2 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        
        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(66, 32, kernel_size=3, padding=1),  # 64 + 2
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )
        
        # Scaling factor to compensate for small spike accumulation values
        self.flow_scale = 20.0
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SpikingFlowNet
        
        Args:
            x: Input tensor [B, C, H, W] - typically stacked event frames or images
            
        Returns:
            Dictionary containing:
                - 'flow': Final optical flow prediction [B, 2, H, W]
                - 'flow_pyramid': Multi-scale flow predictions
                - 'spike_stats': Statistics about spike activity
        """
        batch_size, _, height, width = x.shape
        
        # Storage for membrane potentials across timesteps
        mem_enc1, mem_enc2, mem_enc3, mem_enc4, mem_enc5 = None, None, None, None, None
        mem_dec5, mem_dec4, mem_dec3, mem_dec2 = None, None, None, None
        
        # Accumulate spikes over timesteps
        spike_accumulator = {
            'enc1': torch.zeros(batch_size, 64, height//2, width//2, device=x.device),
            'enc2': torch.zeros(batch_size, 128, height//4, width//4, device=x.device),
            'enc3': torch.zeros(batch_size, 256, height//8, width//8, device=x.device),
            'enc4': torch.zeros(batch_size, 512, height//16, width//16, device=x.device),
            'enc5': torch.zeros(batch_size, 512, height//32, width//32, device=x.device),
        }
        
        total_spikes = 0
        
        # Simulate over multiple timesteps
        for t in range(self.num_timesteps):
            # Encoder
            s1, mem_enc1 = self.encoder1(x, mem_enc1)
            s2, mem_enc2 = self.encoder2(s1, mem_enc2)
            s3, mem_enc3 = self.encoder3(s2, mem_enc3)
            s4, mem_enc4 = self.encoder4(s3, mem_enc4)
            s5, mem_enc5 = self.encoder5(s4, mem_enc5)
            
            # Accumulate spikes
            spike_accumulator['enc1'] += s1
            spike_accumulator['enc2'] += s2
            spike_accumulator['enc3'] += s3
            spike_accumulator['enc4'] += s4
            spike_accumulator['enc5'] += s5
            
            total_spikes += (s1.sum() + s2.sum() + s3.sum() + s4.sum() + s5.sum()).item()
        
        # Decoder (use accumulated spikes)
        d5, mem_dec5 = self.decoder5(spike_accumulator['enc5'], mem_dec5)
        d5_cat = torch.cat([d5, spike_accumulator['enc4']], dim=1)
        
        d4, mem_dec4 = self.decoder4(d5_cat, mem_dec4)
        d4_cat = torch.cat([d4, spike_accumulator['enc3']], dim=1)
        
        d3, mem_dec3 = self.decoder3(d4_cat, mem_dec3)
        d3_cat = torch.cat([d3, spike_accumulator['enc2']], dim=1)
        
        d2, mem_dec2 = self.decoder2(d3_cat, mem_dec2)
        
        # Multi-scale flow predictions
        flow5 = self.flow_head5(spike_accumulator['enc5'])
        flow4 = self.flow_head4(d4)
        flow3 = self.flow_head3(d3)
        flow2 = self.flow_head2(d2)
        
        # Upsample and refine
        flow2_up = nn.functional.interpolate(flow2, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final refinement
        refine_input = torch.cat([spike_accumulator['enc1'], flow2_up], dim=1)
        flow_final = self.refine(refine_input)
        
        # Scale flow to match expected magnitudes
        flow_final = flow_final * self.flow_scale
        
        # Upsample to original resolution
        flow_final = nn.functional.interpolate(flow_final, size=(height, width), mode='bilinear', align_corners=False)
        
        # Calculate spike statistics
        total_neurons = sum([s.numel() for s in spike_accumulator.values()]) * self.num_timesteps
        spike_rate = total_spikes / total_neurons if total_neurons > 0 else 0.0
        
        return {
            'flow': flow_final,
            'flow_pyramid': {
                'flow5': flow5,
                'flow4': flow4,
                'flow3': flow3,
                'flow2': flow2,
            },
            'spike_stats': {
                'total_spikes': total_spikes,
                'spike_rate': spike_rate,
                'num_timesteps': self.num_timesteps
            }
        }


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
