import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any
from .quant_utils import *


class BaseLayer(nn.Module):
    def __init__(
        self,
        config: Optional[dict] = None,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        layer_name: str = "base",
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

        self.weights = {}
    
    def log_params(self, x, out):
        if self.enable_logging_params and self.logger is not None and self.forward_count % 100 == 0:
            with torch.no_grad():
                self.logger.log_scalars(f'params/{self.layer_name}/input', {
                    'min': x.min().item(),
                    'max': x.max().item(),
                    'mean': x.mean().item(),
                    'std': x.std().item()
                }, self.forward_count)

                for name, weight in self.weights.items():
                    self.logger.log_scalars(f'params/{self.layer_name}/weights_{name}', {
                        'min': weight.min().item(),
                        'max': weight.max().item(),
                        'mean':weight.mean().item(),
                        'std': weight.std().item()
                    }, self.forward_count)
                    self.logger.log_histogram(f'params/{self.layer_name}/weights_{name}_hist', weight, self.forward_count)

                self.logger.log_scalars(f'params/{self.layer_name}/act', {
                    'min': out.min().item(),
                    'max': out.max().item(),
                    'mean':out.mean().item(),
                    'std': out.std().item()
                }, self.forward_count)
    
class QuantizedLinear(BaseLayer):
    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        bias: bool = False,
        config: Optional[dict] = None,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        layer_name: str = "linear",
    ):
        super().__init__(config=config, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, layer_name=layer_name)
        self.forward_count = 0
        
        self.lin = nn.Linear(
            in_feat, out_feat, bias=bias
        )

        if self.quantize_weights:
            self.weight_quant = QuantWeight()
        
        if self.quantize_activations:
            self.act_quant = QuantAct(
                config=config,
                layer_name=self.layer_name,  
            )
        
        self.weight = {'linear': self.lin.weight}
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_weights and self.weight_bit_width < 32:
            weight = self.weight_quant(self.lin.weight)
        else:
            weight = self.lin.weight

        out = F.linear(
            input=x, 
            weight=weight, 
            bias=self.lin.bias,
        )
        
        if self.quantize_activations and self.act_bit_width < 32:   
            out_act = self.act_quant(out)
        else:
            out_act = out
        
        self.forward_count += 1

        self.log_params(x, out_act) 

        return out_act

class QuantizedConv2d(BaseLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        s: int = 1,
        p: int = 0,
        groups: int = 1,
        use_norm = None,
        use_bias = None,
        config: Optional[dict] = None,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        layer_name: str = "conv",
    ):
        super().__init__(config=config, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, layer_name=layer_name)

        self.forward_count = 0
        self.use_norm = config.get('use_norm', False) if use_norm is None else use_norm
        self.use_bias = config.get('use_bias', False) if use_bias is None else use_bias
        self.groups = groups
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, k,
            s, p, groups=self.groups, bias=self.use_bias
        )

        if self.use_norm: 
            self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=True)

        if self.quantize_weights:
            self.weight_quant = QuantWeight()
        
        if self.quantize_activations:
            self.act_quant = QuantAct(
                config=config,
                layer_name=self.layer_name,  
            )
    
        self.weights = {'conv': self.conv.weight}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_weights and self.weight_bit_width < 32:
            weight = self.weight_quant(self.conv.weight)
        else:
            weight = self.conv.weight
        
        out = F.conv2d(
            input=x, 
            weight=weight, 
            bias=self.conv.bias,
            groups=self.conv.groups,
            stride=self.conv.stride,
            padding=self.conv.padding,
        )

        if self.use_norm:
            out = self.norm(out)
        
        if self.quantize_activations and self.act_bit_width < 32:   
            out_act = self.act_quant(out)
        else:
            out_act = out
        
        self.forward_count += 1

        self.log_params(x, out_act)
        
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
        self.reset = config.get('reset', 0.0)
        self.quantize_mem = config.get('quantize_mem', False)
        self.mem_bit_width = config.get('mem_bit_width', 16)

        self.mem = None

    def forward(self, x, mem) -> Any:
        if mem is None:
            mem = torch.zeros_like(x)

        mem = mem + x

        spk = SurrogateSpike.apply(mem - self.threshold, self.alpha)
        mem = mem * (self.threshold - spk) + self.reset * spk
    
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
        self.reset = config.get('reset', 1.0)
        self.quantize_mem = config.get('quantize_mem', False)
        self.mem_bit_width = config.get('mem_bit_width', 16)

        self.mem = None

    def forward(self, x, mem) -> Any:
        if mem is None:
            mem = torch.zeros_like(x)

        decay_factor = torch.tensor(self.decay, device=x.device, dtype=x.dtype)
        mem = mem * decay_factor + x

        spk = SurrogateSpike.apply(mem - self.threshold, self.alpha)
        mem = mem * (self.threshold - spk) + self.reset * spk
    
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
