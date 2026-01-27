import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any
from .quant_utils import *


class QuantizedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        s: int = 1,
        p: int = 0,
        bias: bool = False,
        config: Optional[dict] = None,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        layer_name: str = "conv",
    ):
        super().__init__()
        self.layer_name = layer_name
        
        # Separate bit-widths for weights and activations
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width

        self.quantize_weights = config.get('quantize_weights', False)
        self.quantize_activations = config.get('quantize_activations', False)

        self.enable_logging_params = config.get('enable_logging_params', False)
        self.logger = None

        self.forward_count = 0
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, k,
            s, p, bias=bias
        )

        if self.quantize_weights:
            self.weight_quant = QuantWeight()
        
        if self.quantize_activations:
            self.act_quant = QuantAct(
                config=config,
                layer_name=self.layer_name,  
            )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights and activations
        """

        if self.quantize_weights and self.weight_bit_width < 32:
            weight = self.weight_quant(self.conv.weight)
        else:
            weight = self.conv.weight
        
        out = F.conv2d(
            input=x, 
            weight=weight, 
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
        )
        
        if self.quantize_activations and self.act_bit_width < 32:   
            out_act = self.act_quant(out)
        else:
            out_act = out
        
        self.forward_count += 1

        if self.enable_logging_params and self.logger is not None and self.forward_count % 100 == 0:
            with torch.no_grad():
                self.logger.log_scalars(f'params/{self.layer_name}/layer/input', {
                    'min': x.min().item(),
                    'max': x.max().item(),
                    'mean': x.mean().item(),
                    'std': x.std().item()
                }, self.forward_count)

                self.logger.log_scalars(f'params/{self.layer_name}/weights', {
                    'min': weight.min().item(),
                    'max': weight.max().item(),
                    'mean': weight.mean().item(),
                    'std': weight.std().item()
                }, self.forward_count)
                self.logger.log_histogram(f'params/{self.layer_name}/weights_hist', weight, self.forward_count)

                self.logger.log_scalars(f'params/{self.layer_name}/weights', {
                    'min': out_act.min().item(),
                    'max': out_act.max().item(),
                    'mean': out_act.mean().item(),
                    'std': out_act.std().item()
                }, self.forward_count)
        
        return out_act


class QuantizedIF(nn.Module):
    def __init__(
        self,
        config: Optional[dict] = None,
        layer_name: str = "if",
    ):
        super().__init__()
        self.threshold = config.get('threshold', 1.0)
        self.alpha = config.get('alpha', 10.0)
        self.quantize_mem = config.get('quantize_mem', False)
        self.mem_bit_width = config.get('mem_bit_width', 16)

        self.mem = None

    def forward(self, x, mem) -> Any:
        if mem is None:
            mem = torch.zeros_like(x)

        mem = mem + x

        spk = SurrogateSpike.apply(mem - self.threshold, self.alpha)
        mem = mem * (1.0 - spk)
    
        # Quantize membrane again after reset if enabled
        if self.quantize_mem:
            mem = QuantMembrane(mem, bit_width=self.mem_bit_width, mem_range=self.threshold * 2.0)
    
        return spk, mem

class QuantizedLIF(nn.Module):
    def __init__(
        self,
        config: Optional[dict] = None,
        layer_name: str = "lif",
    ):
        super().__init__()
        
        self.threshold = config.get('threshold', 1.0)
        self.decay = config.get('decay', 0.5)
        self.alpha = config.get('alpha', 10.0)
        self.quantize_mem = config.get('quantize_mem', False)
        self.mem_bit_width = config.get('mem_bit_width', 16)

        self.mem = None

    def forward(self, x, mem) -> Any:
        if mem is None:
            mem = torch.zeros_like(x)

        decay_factor = torch.tensor(self.decay, device=x.device, dtype=x.dtype)
        mem = mem * decay_factor + x

        spk = SurrogateSpike.apply(mem - self.threshold, self.alpha)
        mem = mem * (1.0 - spk)
    
        # Quantize membrane again after reset if enabled
        if self.quantize_mem:
            mem = QuantMembrane(mem, bit_width=self.mem_bit_width, mem_range=self.threshold * 2.0)
    
        return spk, mem

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
        s = torch.sigmoid(alpha * x)
        grad = alpha * s * (1 - s)
        return grad_out * grad, None
