import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.round()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BinaryActivation(torch.autograd.Function):
    """
    Binary activation function with STE
    Forward: Sign function
    Backward: Straight-through or clipped gradient
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Clip gradient to [-1, 1] region
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1.0] = 0
        return grad_input


class BinaryWeight(torch.autograd.Function):
    """
    Binary weight quantization with scaling
    Uses mean absolute value as scaling factor
    """
    @staticmethod
    def forward(ctx, weight):
        # Calculate scaling factor (mean absolute value per filter)
        alpha = weight.abs().mean(dim=[1, 2, 3], keepdim=True)
        # Binarize
        binary_weight = torch.sign(weight)
        # Scale
        scaled_weight = alpha * binary_weight
        return scaled_weight
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SymmetricQuant(torch.autograd.Function):
    # Use symmetric x quantization, no zero-point
    @staticmethod
    def forward(ctx, x, bit_width, scale=None, zero_point=None, type="max", per_channel=False):
        # Shape: [out_ch, in_ch, kH, kW] 

        qmax = 2 ** (bit_width - 1) - 1

        if scale is None:

            if type == "max":
                if per_channel:
                    # Calculate per-channel max absolute value
                    x_flat = x.view(x.size(0), -1)  # [out_ch, in_ch*kH*kW]
                    scale_val = torch.max(x_flat.abs(), dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)  # [out_ch, 1, 1, 1]
                else:
                    max_abs = x.abs().max()
                    scale_val = max_abs

            elif type == "mse":
                pass

            scale = scale_val / qmax

        # Quantize then dequantize
        x_q = x / scale
        x_q = torch.clamp(x_q, -qmax-1, qmax)
        x_q = x_q.round()

        return x_q, scale, None
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None

class AsymmetricQuant(torch.autograd.Function):
    # Use asymmetric quantization, zero-point
    @staticmethod
    def forward(ctx, x, bit_width, scale=None, zero_point=None, type="max", per_channel=False):
        # Shape: [out_ch, in_ch, kH, kW] 

        qmax = 2 ** bit_width - 1


        if type == "max":
            if per_channel:
                # Calculate per-channel max absolute value
                x_flat = x.view(x.size(0), -1)  # [out_ch, in_ch*kH*kW]
                max_val = torch.max(x_flat, dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)  # [out_ch, 1, 1, 1]
                min_val = torch.min(x_flat, dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)  # [out_ch, 1, 1, 1]
                scale_val = max_val - min_val

                zero_point_calc = torch.min(x_flat, dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)  # [out_ch, 1, 1, 1]
            else:
                max_val = torch.max(x)  
                min_val = torch.min(x)  

                scale_val = max_val - min_val
                zero_point_calc = torch.min(x)

        elif type == "mse":
            pass

        scale_calc = scale_val / qmax

        # Quantize then dequantize
        if scale is None:
            scale = scale_calc
            zero_point = zero_point_calc

        x_q = x / scale + zero_point
        x_q = torch.clamp(x_q, -qmax-1, qmax)
        x_q = x_q.round()

        # Return new calculations
        return x_q, scale_calc, zero_point_calc
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None

class QuantAct(nn.Module):
    def __init__(
        self,
        bit_width: int = 8,
        symmetric: bool = True,
        layer_name: str = "qact",
        quantize: bool = True  
    ):
        super().__init__()
        
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.layer_name = layer_name
        self.quantize = quantize
        self.forward_count = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_q, s, z = AsymmetricQuant.apply(x)

        return x_q, s, z

class QuantWeight(nn.Module):
    def __init__(
        self,
        bit_width: int = 8,
        symmetric: bool = True,
        layer_name: str = "qweight",
    ):
        super().__init__()
        
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.layer_name = layer_name
        self.forward_count = 0
        
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            w_q, s, z = SymmetricQuant.apply(w)
            
    
        return w_q, s, z

def QuantMembrane():
    pass