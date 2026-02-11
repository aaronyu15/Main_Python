import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .snn_layers import *
import torch.nn.functional as F


layers = {
    "SpikingConvBlock": SpikingConvBlock,
    "ConvBlock": ConvBlock,
    "SpikingLinearBlock": SpikingLinearBlock,
}

class EventSNNFlowNetLite(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.config = config
        
        # Patch mode configuration
        self.patch_mode = config.get("patch_mode", False)
        self.patch_size = config.get("patch_size", 64)

        self.quantize_weights = config.get("quantize_weights", False)
        self.quantize_activations = config.get("quantize_activations", False)
        self.quantize_mem = config.get("quantize_mem", False)

        self.output_bit_width = config.get("output_bit_width", 8)
        self.input_bit_width = config.get("input_bit_width", 8)
        self.base_ch = config.get("base_ch", 32)
        self.use_polarity = config.get("use_polarity", True)

        self.logger = None  # TensorBoard logger
        self.disable_skip = False
        
        conv_layer = layers[config.get("conv_type", "SpikingConvBlock")]

        self.e1 = SpikingConvBlock(
            2 if self.use_polarity else 1,
            self.base_ch,
            k=3,
            s=2,
            p=1,
            config=config,
            weight_bit_width=self.input_bit_width,
            act_bit_width=self.input_bit_width,
            layer_name="e1",
        ) # -> 32x32

        self.e2 = conv_layer(
            self.base_ch, 
            self.base_ch*2, 
            k=3, 
            s=2, 
            p=1, 
            config=config,
            layer_name="e2"
        ) # -> 16x16

        self.e3 = conv_layer(
            self.base_ch*2, 
            self.base_ch*2, 
            k=3, 
            s=1, 
            p=1, 
            config=config,
            layer_name="e3"
        ) # -> 16x16

        #self.e4 = conv_layer(
        #    self.base_ch*2, 
        #    self.base_ch*2, 
        #    k=3, 
        #    s=1, 
        #    p=1, 
        #    config=config,
        #    layer_name="e4"
        #) # -> 16x16

        #self.d4 = conv_layer(
        #    self.base_ch*2,
        #    self.base_ch*2,
        #    k=3,
        #    s=1,
        #    p=1,
        #    groups=2,
        #    config=config,
        #    layer_name="d4",
        #) # -> 32x32

        self.d3 = conv_layer(
            self.base_ch*2,
            self.base_ch*2,
            k=3,
            s=1,
            p=1,
            groups=2,
            config=config,
            layer_name="d3",
        ) # -> 32x32

        self.d2 = conv_layer(
            self.base_ch*2,
            self.base_ch,
            k=3,
            s=1,
            p=1,
            groups=2,
            config=config,
            layer_name="d2",
        ) # -> 32x32
            
        self.d1 = conv_layer(
            self.base_ch,
            self.base_ch,
            k=3,
            s=1,
            p=1,
            groups=2,
            config=config,
            layer_name="d1",
        ) # -> 64x64
            
        # Flow prediction head - predicts flow at each pixel in patch
        self.flow_head = ConvBlock(
            self.base_ch,
            2,
            k=3,
            s=1,
            p=1,
            groups=2,
            use_norm=False,
            use_bias=False,
            config=config,
            weight_bit_width=self.output_bit_width,
            act_bit_width=self.output_bit_width,
            layer_name="flow_head",
        ) # -> [N, 2, 64, 64]
            


    def set_logger(self, logger):
        self.logger = logger
        for module in self.modules():
            if hasattr(module, "logger"):
                module.logger = logger

 

    def forward(self, x):
        """Forward pass for full-image processing (original)"""
        N, T, C, H, W = x.shape
        if self.use_polarity:
            assert C == 2, "Expected 2 polarity channels"
        else:
            assert C == 1, "Expected 1 channel when not using polarity"

        mem_e1 = mem_e2 = mem_e3 = mem_e4 = None
        mem_d4 = mem_d3 = mem_d2 = mem_d1 = None

        flow_acc = None  # [N,2,H,W]

        for t in range(T):
            xt = x[:, t]  # [N,2,H,W]

            s1, mem_e1 = self.e1(xt, mem_e1)  # [N, base,   H/2, W/2]
            s2, mem_e2 = self.e2(s1, mem_e2)  # [N, 2base,  H/4, W/4]
            s3, mem_e3 = self.e3(s2, mem_e3) 
            #s4, mem_e4 = self.e4(s3, mem_e4)  

            #d4, mem_d4 = self.d4(s4, mem_d4)
            #d4 = d4 + s3 if self.disable_skip is False else d3

            d3, mem_d3 = self.d3(s3, mem_d3)
            d3 = d3 + s2 if self.disable_skip is False else d3

            d2, mem_d2 = self.d2(d3, mem_d2)
            d2 = F.interpolate(d2, scale_factor=2, mode="nearest")  # -> H/2
            d2 = d2 + s1 if self.disable_skip is False else d2                                                                  

            d1, mem_d1 = self.d1(d2, mem_d1)
            mem_d1_i = F.interpolate(mem_d1, scale_factor=2, mode="nearest")  # -> H

            dflow = self.flow_head(mem_d1_i)

            if flow_acc is None:
                flow_acc = dflow
            else:
                flow_acc = flow_acc + dflow
        
        return {"flow": flow_acc}

