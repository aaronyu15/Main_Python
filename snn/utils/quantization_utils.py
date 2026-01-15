"""
Quantization Utilities for Model Conversion and Analysis

Functions for:
- Converting full-precision models to quantized models
- Analyzing quantization impact
- Exporting quantized models for hardware deployment
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


def enable_quantization_in_model(
    model: nn.Module,
    bit_width: int = 8,
    verbose: bool = True
) -> nn.Module:
    """
    Enable quantization in a model by adding quantization layers
    
    This is useful when you have a pre-trained model without quantization
    and want to add quantization layers for QAT fine-tuning.
    
    Args:
        model: Model to enable quantization in
        bit_width: Target bit-width for quantization
        verbose: Print information about changes
        
    Returns:
        Model with quantization enabled
    """
    from ..quantization import QuantizationAwareLayer
    
    count = 0
    for name, module in model.named_modules():
        # Check if module has quantization support but it's disabled
        if hasattr(module, 'quantize') and hasattr(module, 'bit_width'):
            if not module.quantize:
                module.quantize = True
                module.bit_width = bit_width
                # Add quantization layer
                module.quant_layer = QuantizationAwareLayer(bit_width=bit_width)
                count += 1
            else:
                # Update bit-width if quantization already enabled
                module.bit_width = bit_width
                if hasattr(module, 'quant_layer') and module.quant_layer is not None:
                    module.quant_layer.bit_width = bit_width
                    # Update qmin/qmax
                    if module.quant_layer.symmetric:
                        module.quant_layer.qmin = -(2 ** (bit_width - 1))
                        module.quant_layer.qmax = 2 ** (bit_width - 1) - 1
                    else:
                        module.quant_layer.qmin = 0
                        module.quant_layer.qmax = 2 ** bit_width - 1
    
    if verbose:
        print(f"Enabled quantization in {count} layers with {bit_width}-bit precision")
    
    return model


def disable_quantization_in_model(model: nn.Module, verbose: bool = True) -> nn.Module:
    """
    Disable quantization in a model (useful for comparison)
    
    Args:
        model: Model to disable quantization in
        verbose: Print information
        
    Returns:
        Model with quantization disabled
    """
    count = 0
    for module in model.modules():
        if hasattr(module, 'quantize'):
            if module.quantize:
                module.quantize = False
                count += 1
    
    if verbose:
        print(f"Disabled quantization in {count} layers")
    
    return model


def count_quantized_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters by quantization bit-width
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with parameter counts per bit-width
    """
    stats = {
        'total': 0,
        'full_precision': 0,  # No quantization
        '8bit': 0,
        '4bit': 0,
        '2bit': 0,
        '1bit': 0,
        'other': 0
    }
    
    for module in model.modules():
        if hasattr(module, 'quantize') and module.quantize:
            # Get parameters from this module
            params = sum(p.numel() for p in module.parameters(recurse=False))
            bit_width = getattr(module, 'bit_width', 32)
            
            stats['total'] += params
            
            if bit_width == 8:
                stats['8bit'] += params
            elif bit_width == 4:
                stats['4bit'] += params
            elif bit_width == 2:
                stats['2bit'] += params
            elif bit_width == 1:
                stats['1bit'] += params
            else:
                stats['other'] += params
        else:
            # Full precision
            params = sum(p.numel() for p in module.parameters(recurse=False))
            stats['total'] += params
            stats['full_precision'] += params
    
    return stats


def estimate_model_size(model: nn.Module, verbose: bool = True) -> Dict[str, float]:
    """
    Estimate model size in MB for different precision levels
    
    Args:
        model: Model to analyze
        verbose: Print results
        
    Returns:
        Dictionary with size estimates
    """
    param_stats = count_quantized_parameters(model)
    
    # Calculate size in MB
    # Full precision: 4 bytes per param (FP32)
    # 8-bit: 1 byte per param
    # 4-bit: 0.5 bytes per param
    # 2-bit: 0.25 bytes per param
    # 1-bit: 0.125 bytes per param
    
    size_mb = {
        'full_precision': param_stats['full_precision'] * 4 / (1024 ** 2),
        '8bit': param_stats['8bit'] * 1 / (1024 ** 2),
        '4bit': param_stats['4bit'] * 0.5 / (1024 ** 2),
        '2bit': param_stats['2bit'] * 0.25 / (1024 ** 2),
        '1bit': param_stats['1bit'] * 0.125 / (1024 ** 2),
        'other': param_stats['other'] * 4 / (1024 ** 2)  # Assume FP32
    }
    
    size_mb['total'] = sum(size_mb.values())
    
    if verbose:
        print(f"\nModel Size Estimation:")
        print(f"  Total parameters: {param_stats['total']:,}")
        print(f"  Estimated size: {size_mb['total']:.2f} MB")
        print(f"    - Full precision: {size_mb['full_precision']:.2f} MB ({param_stats['full_precision']:,} params)")
        print(f"    - 8-bit: {size_mb['8bit']:.2f} MB ({param_stats['8bit']:,} params)")
        print(f"    - 4-bit: {size_mb['4bit']:.2f} MB ({param_stats['4bit']:,} params)")
        print(f"    - 2-bit: {size_mb['2bit']:.2f} MB ({param_stats['2bit']:,} params)")
        print(f"    - 1-bit: {size_mb['1bit']:.2f} MB ({param_stats['1bit']:,} params)")
    
    return size_mb


def compare_model_outputs(
    model1: nn.Module,
    model2: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compare outputs between two models (e.g., full-precision vs quantized)
    
    Args:
        model1: First model (e.g., full-precision)
        model2: Second model (e.g., quantized)
        input_tensor: Input to test with
        device: Device to run on
        
    Returns:
        Dictionary with comparison metrics
    """
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)
        
        # Extract flow predictions
        flow1 = output1['flow']
        flow2 = output2['flow']
        
        # Compute metrics
        mse = torch.mean((flow1 - flow2) ** 2).item()
        mae = torch.mean(torch.abs(flow1 - flow2)).item()
        max_diff = torch.max(torch.abs(flow1 - flow2)).item()
        
        # Relative error
        flow1_norm = torch.norm(flow1)
        if flow1_norm > 0:
            relative_error = torch.norm(flow1 - flow2) / flow1_norm
            relative_error = relative_error.item()
        else:
            relative_error = float('inf')
    
    return {
        'mse': mse,
        'mae': mae,
        'max_diff': max_diff,
        'relative_error': relative_error
    }


def export_quantized_model_info(
    model: nn.Module,
    output_path: str,
    include_weights: bool = False
) -> None:
    """
    Export quantized model information for hardware deployment
    
    Args:
        model: Quantized model
        output_path: Path to save info
        include_weights: Whether to include actual weights (can be large)
    """
    import json
    
    info = {
        'model_type': type(model).__name__,
        'quantization_info': {},
        'layer_info': []
    }
    
    for name, module in model.named_modules():
        if hasattr(module, 'quantize') and module.quantize:
            layer_info = {
                'name': name,
                'type': type(module).__name__,
                'bit_width': getattr(module, 'bit_width', None),
                'num_params': sum(p.numel() for p in module.parameters(recurse=False))
            }
            
            # Add quantization layer statistics if available
            if hasattr(module, 'quant_layer') and module.quant_layer is not None:
                ql = module.quant_layer
                layer_info['quant_stats'] = {
                    'running_min': float(ql.running_min.item()) if hasattr(ql, 'running_min') else None,
                    'running_max': float(ql.running_max.item()) if hasattr(ql, 'running_max') else None,
                    'symmetric': getattr(ql, 'symmetric', None),
                }
            
            info['layer_info'].append(layer_info)
    
    # Summary statistics
    param_stats = count_quantized_parameters(model)
    size_stats = estimate_model_size(model, verbose=False)
    
    info['quantization_info']['param_stats'] = param_stats
    info['quantization_info']['size_mb'] = size_stats
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Exported quantization info to: {output_path}")


def print_quantization_summary(model: nn.Module) -> None:
    """
    Print a summary of quantization in the model
    
    Args:
        model: Model to summarize
    """
    print("\n" + "=" * 80)
    print("Quantization Summary")
    print("=" * 80)
    
    # Count layers by bit-width
    bit_width_counts = {}
    for module in model.modules():
        if hasattr(module, 'quantize') and module.quantize:
            bit_width = getattr(module, 'bit_width', None)
            if bit_width is not None:
                bit_width_counts[bit_width] = bit_width_counts.get(bit_width, 0) + 1
    
    print(f"\nQuantized Layers:")
    for bit_width, count in sorted(bit_width_counts.items()):
        print(f"  {bit_width}-bit: {count} layers")
    
    # Parameter and size estimates
    estimate_model_size(model, verbose=True)
    
    print("=" * 80 + "\n")
