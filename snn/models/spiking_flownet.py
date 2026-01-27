import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .snn_layers import SpikingConvBlock, ConvBlock
import torch.nn.functional as F


class EventSNNFlowNetLite(nn.Module):
    def __init__(
        self,
        config,
        base_ch=32,
        decay=0.5,
        threshold=1.0,
        alpha=10.0,
    ):
        super().__init__()

        self.config = config

        self.quantize_weights = config.get("quantize_weights", False)
        self.quantize_activations = config.get("quantize_activations", False)
        self.quantize_mem = config.get("quantize_mem", False)

        self.output_bit_width = config.get("output_bit_width", 8)
        self.input_bit_width = config.get("input_bit_width", 8)

        self.logger = None  # TensorBoard logger

        # O = (F - K + 2P)/S + 1
        # Input shape: 320x320
        self.e1 = SpikingConvBlock(
            2,
            base_ch,
            k=3,
            s=2,
            p=1,
            config=config,
            weight_bit_width=self.input_bit_width,
            act_bit_width=self.input_bit_width,
            layer_name="e1",
        ) # -> 160x160

        self.e2 = SpikingConvBlock(
            base_ch, 
            base_ch * 2, 
            k=3, 
            s=2, 
            p=1, 
            config=config,
            layer_name="e2"
        ) # -> 80x80

        self.e3 = SpikingConvBlock(
            base_ch * 2, 
            base_ch * 4, 
            k=3, 
            s=2, 
            p=1, 
            config=config,
            layer_name="e3"
        ) # -> 40x40

        self.d3 = SpikingConvBlock(
            base_ch * 4,
            base_ch * 2,
            k=3,
            s=1,
            p=1,
            config=config,
            layer_name="d3",
        ) # -> 80x80

        self.d2 = SpikingConvBlock(
            base_ch * 2,
            base_ch,
            k=3,
            s=1,
            p=1,
            config=config,
            layer_name="d2",
        ) # -> 160x160

        self.d1 = SpikingConvBlock(
            base_ch,
            base_ch,
            k=3,
            s=1,
            p=1,
            config=config,
            layer_name="d1",
        ) # -> 320x320

        self.flow_head = ConvBlock(
            base_ch,
            2,
            k=3,
            s=1,
            p=1,
            config=config,
            weight_bit_width=self.output_bit_width,
            act_bit_width=self.output_bit_width,
            layer_name="flow_head",
        ) # -> 320x320


    def set_logger(self, logger):
        self.logger = logger
        for module in self.modules():
            if hasattr(module, "logger"):
                module.logger = logger

 

    def forward(self, x):

        N, T, C, H, W = x.shape
        assert C == 2, "Expected 2 polarity channels"

        mem_e1 = mem_e2 = mem_e3 = None
        mem_d3 = mem_d2 = mem_d1 = None

        flow_acc = None  # [N,2,H,W]

        for t in range(T):
            xt = x[:, t]  # [N,2,H,W]

            s1, mem_e1 = self.e1(xt, mem_e1)  # [N, base,   H/2, W/2]
            s2, mem_e2 = self.e2(s1, mem_e2)  # [N, 2base,  H/4, W/4]
            s3, mem_e3 = self.e3(s2, mem_e3)  # [N, 4base,  H/8, W/8]

            d3 = F.interpolate(s3, scale_factor=2, mode="nearest")  # -> H/4
            d3, mem_d3 = self.d3(d3, mem_d3)
            d3 = d3 + s2

            d2 = F.interpolate(d3, scale_factor=2, mode="nearest")  # -> H/2
            d2, mem_d2 = self.d2(d2, mem_d2)
            d2 = d2 + s1

            d1 = F.interpolate(d2, scale_factor=2, mode="nearest")  # -> H
            d1, mem_d1 = self.d1(d1, mem_d1)

            dflow = self.flow_head(mem_d1)

            if flow_acc is None:
                flow_acc = dflow
            else:
                flow_acc = flow_acc + dflow
        return {"flow": flow_acc}
