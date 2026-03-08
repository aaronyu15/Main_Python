"""
Simplified Quantization Utilities — PTQ and QAT

Two modes:
  PTQ (Post-Training Quantization):
    - Load pretrained model, run calibration pass over data
    - Scales computed from observed min/max (or percentile) statistics
    - No training required

  QAT (Quantization-Aware Training):
    - Fake quantization in forward pass with STE (straight-through estimator)
    - Weight scales derived from weight tensor each forward: scale = max(|w|) / qmax
    - Activation scales tracked via EMA (exponential moving average) of running min/max
    - Fine-tune the pretrained model (weights update, scales follow automatically)

Scale granularity:
  per_channel: one scale per output channel (best accuracy)
  per_layer:   one scale per layer (simpler)
  global:      single scale shared across entire network (simplest for FPGA)
"""

import torch
import torch.nn as nn
import warnings
import math
from typing import Optional, Dict, Tuple

# ============================================================================
# Warning infrastructure
# ============================================================================

_QUANT_RANGE_WARNINGS_ENABLED = True
_QUANT_WARNING_COUNTS = {}
_QUANT_WARNING_LIMIT = 10


def enable_quantization_warnings(enabled: bool = True):
    global _QUANT_RANGE_WARNINGS_ENABLED
    _QUANT_RANGE_WARNINGS_ENABLED = enabled


def reset_quantization_warning_counts():
    global _QUANT_WARNING_COUNTS
    _QUANT_WARNING_COUNTS = {}


def _warn_if_clipped(tensor: torch.Tensor, qmin: int, qmax: int,
                     layer_name: str, tensor_type: str):
    """Warn if values fall outside quantization range."""
    if not _QUANT_RANGE_WARNINGS_ENABLED:
        return
    global _QUANT_WARNING_COUNTS
    key = f"{layer_name}_{tensor_type}"
    count = _QUANT_WARNING_COUNTS.get(key, 0)
    if count >= _QUANT_WARNING_LIMIT:
        return
    with torch.no_grad():
        out_of_range = ((tensor < qmin) | (tensor > qmax)).sum().item()
        if out_of_range > 0:
            pct = 100.0 * out_of_range / tensor.numel()
            msg = (f"[Quant] {layer_name}/{tensor_type}: "
                   f"{out_of_range}/{tensor.numel()} ({pct:.2f}%) clipped "
                   f"to [{qmin}, {qmax}]. "
                   f"Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
            if count + 1 == _QUANT_WARNING_LIMIT:
                msg += " (suppressing further warnings)"
            warnings.warn(msg, RuntimeWarning)
            _QUANT_WARNING_COUNTS[key] = count + 1


# ============================================================================
# Core fake-quantize (STE)
# ============================================================================

def fake_quantize_symmetric(x: torch.Tensor, scale: torch.Tensor,
                            bit_width: int, layer_name: str = "",
                            tensor_type: str = "value") -> torch.Tensor:
    """
    Symmetric fake quantization with straight-through estimator.

    Forward: quantize -> dequantize (introduces quantization noise)
    Backward: gradient passes straight through (STE)

    Args:
        x: input tensor
        scale: scale factor (can be scalar, per-channel, etc.)
        bit_width: number of bits
        layer_name: for warning messages
        tensor_type: "weight" or "activation"
    """
    qmax = 2 ** (bit_width - 1) - 1
    qmin = -qmax - 1
    scale = scale.abs().clamp(min=1e-8)

    # Quantize
    x_scaled = x / scale
    # STE: round in forward, pass gradient through
    x_int = x_scaled + (x_scaled.round() - x_scaled).detach()

    # Warn about clipping
    _warn_if_clipped(x_int, qmin, qmax, layer_name, tensor_type)

    # Clamp
    x_clamped = x_int.clamp(qmin, qmax)

    # Dequantize
    return x_clamped * scale


def fake_quantize_asymmetric(x: torch.Tensor, scale: torch.Tensor,
                             zero_point: torch.Tensor, bit_width: int,
                             layer_name: str = "",
                             tensor_type: str = "value") -> torch.Tensor:
    """
    Asymmetric fake quantization with STE.
    Useful for activations with non-symmetric ranges (e.g., ReLU outputs).
    """
    qmax = 2 ** bit_width - 1
    qmin = 0
    scale = scale.abs().clamp(min=1e-8)

    x_scaled = x / scale + zero_point
    x_int = x_scaled + (x_scaled.round() - x_scaled).detach()

    _warn_if_clipped(x_int, qmin, qmax, layer_name, tensor_type)

    x_clamped = x_int.clamp(qmin, qmax)
    return (x_clamped - zero_point) * scale


# ============================================================================
# Scale computation helpers
# ============================================================================

def compute_weight_scale(weight: torch.Tensor, bit_width: int,
                         scale_type: str = "per_channel") -> torch.Tensor:
    """
    Compute weight scale from weight statistics.

    scale = max(|w|) / qmax

    This is recomputed every forward pass in QAT -- no learned parameter needed.
    """
    qmax = 2 ** (bit_width - 1) - 1

    if scale_type == "per_channel":
        # Per output channel: flatten spatial dims, take max abs per channel
        w_flat = weight.view(weight.size(0), -1)
        max_abs = w_flat.abs().max(dim=1)[0]  # shape: [out_channels]
        scale = (max_abs / qmax).clamp(min=1e-8)
        return scale
    elif scale_type == "per_layer":
        max_abs = weight.abs().max()
        return (max_abs / qmax).clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")


def compute_act_scale_minmax(min_val: torch.Tensor, max_val: torch.Tensor,
                             bit_width: int, symmetric: bool = True
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute activation scale/zero_point from observed min/max.

    Returns:
        (scale, zero_point)
    """
    if symmetric:
        qmax = 2 ** (bit_width - 1) - 1
        abs_max = torch.max(min_val.abs(), max_val.abs())
        scale = (abs_max / qmax).clamp(min=1e-8)
        zero_point = torch.zeros_like(scale)
    else:
        qmax = 2 ** bit_width - 1
        scale = ((max_val - min_val) / qmax).clamp(min=1e-8)
        zero_point = (-min_val / scale).round()
    return scale, zero_point


# ============================================================================
# Global Scale (for entire-network single scale)
# ============================================================================

class GlobalScale:
    """
    Holds a single scale value shared across all layers in the network.

    Usage:
        GlobalScale.reset()
        GlobalScale.set_weight_scale(value)
        s = GlobalScale.get_weight_scale()
    """
    _weight_scale: Optional[torch.Tensor] = None
    _act_scale: Optional[torch.Tensor] = None
    _act_zero_point: Optional[torch.Tensor] = None

    @classmethod
    def reset(cls):
        cls._weight_scale = None
        cls._act_scale = None
        cls._act_zero_point = None

    @classmethod
    def set_weight_scale(cls, scale: torch.Tensor):
        cls._weight_scale = scale.detach()

    @classmethod
    def set_act_scale(cls, scale: torch.Tensor,
                      zero_point: Optional[torch.Tensor] = None):
        cls._act_scale = scale.detach()
        cls._act_zero_point = (zero_point.detach()
                               if zero_point is not None
                               else torch.tensor(0.0))

    @classmethod
    def get_weight_scale(cls) -> Optional[torch.Tensor]:
        return cls._weight_scale

    @classmethod
    def get_act_scale(cls) -> Optional[torch.Tensor]:
        return cls._act_scale

    @classmethod
    def get_act_zero_point(cls) -> Optional[torch.Tensor]:
        return (cls._act_zero_point
                if cls._act_zero_point is not None
                else torch.tensor(0.0))


# ============================================================================
# QuantWeight -- weight quantizer
# ============================================================================

class QuantWeight(nn.Module):
    """
    Weight fake-quantizer.

    PTQ mode: scale is computed once during calibration and frozen (buffer).
    QAT mode: scale is recomputed from weight statistics every forward pass.

    Args:
        bit_width: quantization bit width
        layer_name: for debug messages
        num_channels: output channels (needed for per_channel)
        scale_type: "per_channel" | "per_layer" | "global"
        config: config dict (overrides other args)
    """
    def __init__(self, bit_width: int = 8, layer_name: str = "qw",
                 num_channels: int = 1, scale_type: str = None,
                 config: Optional[dict] = None):
        super().__init__()

        self.bit_width = (config.get('weight_bit_width', bit_width)
                          if config else bit_width)
        self.layer_name = layer_name

        # Determine scale type
        if scale_type is not None:
            self.scale_type = scale_type
        elif config is not None:
            self.scale_type = config.get('weight_scale_type', 'per_channel')
        else:
            self.scale_type = 'per_channel'

        self.num_channels = num_channels

        # Quant mode: "qat" (default) or "ptq"
        quant_mode = config.get('quant_mode', 'qat') if config else 'qat'
        self._ptq_mode = (quant_mode == 'ptq')

        # Scale stored as buffer (not a parameter -- not learned)
        if self.scale_type == 'per_channel' and num_channels > 1:
            self.register_buffer('scale', torch.ones(num_channels))
        else:
            self.register_buffer('scale', torch.ones(1))

        self._calibrated = False

    def calibrate(self, weight: torch.Tensor):
        """Compute and freeze scale from weight tensor (used in PTQ)."""
        with torch.no_grad():
            if self.scale_type == 'global':
                s = GlobalScale.get_weight_scale()
                if s is not None:
                    self.scale.copy_(s.to(self.scale.device))
                else:
                    s = compute_weight_scale(weight, self.bit_width,
                                             'per_layer')
                    self.scale.copy_(s)
            else:
                s = compute_weight_scale(weight, self.bit_width,
                                         self.scale_type)
                self.scale.copy_(s.to(self.scale.device))
        self._calibrated = True

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self._ptq_mode:
            # PTQ: use frozen calibrated scale
            if not self._calibrated:
                self.calibrate(weight)
            scale = self.scale
        else:
            # QAT: recompute scale from current weights each forward pass
            if self.scale_type == 'global':
                s = GlobalScale.get_weight_scale()
                if s is not None:
                    scale = s.to(weight.device)
                else:
                    scale = compute_weight_scale(weight, self.bit_width,
                                                 'per_layer')
            else:
                scale = compute_weight_scale(weight, self.bit_width,
                                             self.scale_type)
            # Store for export
            with torch.no_grad():
                self.scale.copy_(scale)

        # Reshape for broadcasting
        if scale.dim() == 1 and weight.dim() > 1:
            shape = [scale.size(0)] + [1] * (weight.dim() - 1)
            scale = scale.view(*shape)

        return fake_quantize_symmetric(weight, scale, self.bit_width,
                                       self.layer_name, "weight")

    def get_int_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Get integer-quantized weights for export."""
        scale = self.scale.clone()
        if scale.dim() == 1 and weight.dim() > 1:
            shape = [scale.size(0)] + [1] * (weight.dim() - 1)
            scale = scale.view(*shape)
        scale = scale.abs().clamp(min=1e-8)
        qmax = 2 ** (self.bit_width - 1) - 1
        return (weight / scale).round().clamp(-qmax - 1, qmax).to(
            torch.int8 if self.bit_width <= 8 else torch.int16)

    def extra_repr(self) -> str:
        mode = "ptq" if self._ptq_mode else "qat"
        return (f'bit_width={self.bit_width}, scale_type={self.scale_type}, '
                f'mode={mode}')


# ============================================================================
# QuantAct -- activation quantizer
# ============================================================================

class QuantAct(nn.Module):
    """
    Activation fake-quantizer.

    PTQ mode: scale computed during calibration from observed min/max.
    QAT mode: scale tracked via EMA of running min/max, fake-quantized with STE.

    Args:
        bit_width: quantization bit width
        symmetric: use symmetric quantization (default True for SNN spikes)
        layer_name: for debug messages
        num_channels: output channels (for per_channel mode)
        scale_type: "per_channel" | "per_layer" | "global"
        ema_decay: decay factor for EMA of running min/max (QAT mode)
        config: config dict (overrides other args)
    """
    def __init__(self, bit_width: int = 8, symmetric: bool = True,
                 layer_name: str = "qa", num_channels: int = 1,
                 scale_type: str = None, ema_decay: float = 0.999,
                 config: Optional[dict] = None):
        super().__init__()

        self.bit_width = (config.get('act_bit_width', bit_width)
                          if config else bit_width)
        self.symmetric = (config.get('act_symmetric', symmetric)
                          if config else symmetric)
        self.layer_name = layer_name
        self.ema_decay = ema_decay

        # Determine scale type
        if scale_type is not None:
            self.scale_type = scale_type
        elif config is not None:
            self.scale_type = config.get('act_scale_type', 'per_layer')
        else:
            self.scale_type = 'per_layer'

        self.num_channels = num_channels

        quant_mode = config.get('quant_mode', 'qat') if config else 'qat'
        self._ptq_mode = (quant_mode == 'ptq')

        # Buffers for scale, zero_point, and running stats
        if self.scale_type == 'per_channel' and num_channels > 1:
            self.register_buffer('scale', torch.ones(num_channels))
            self.register_buffer('zero_point', torch.zeros(num_channels))
            self.register_buffer('running_min', torch.zeros(num_channels))
            self.register_buffer('running_max', torch.ones(num_channels))
        else:
            self.register_buffer('scale', torch.ones(1))
            self.register_buffer('zero_point', torch.zeros(1))
            self.register_buffer('running_min', torch.zeros(1))
            self.register_buffer('running_max', torch.ones(1))

        self._calibrated = False
        self._calibrating = False
        self._num_batches_seen = 0

    def _observe(self, x: torch.Tensor):
        """Update running min/max from observed activations."""
        with torch.no_grad():
            if (self.scale_type == 'per_channel'
                    and self.num_channels > 1 and x.dim() > 1):
                # Per-channel min/max: x shape [B, C, H, W] or [B, C]
                dims = [i for i in range(x.dim()) if i != 1]
                cur_min = x.amin(dim=dims)
                cur_max = x.amax(dim=dims)
            else:
                cur_min = x.min().view_as(self.running_min)
                cur_max = x.max().view_as(self.running_max)

            if self._num_batches_seen == 0:
                self.running_min.copy_(cur_min)
                self.running_max.copy_(cur_max)
            else:
                d = self.ema_decay
                self.running_min.mul_(d).add_(cur_min * (1 - d))
                self.running_max.mul_(d).add_(cur_max * (1 - d))

            self._num_batches_seen += 1

    def _update_scale(self):
        """Recompute scale/zero_point from running stats."""
        with torch.no_grad():
            if self.scale_type == 'global':
                gs = GlobalScale.get_act_scale()
                if gs is not None:
                    self.scale.copy_(gs.to(self.scale.device))
                    gzp = GlobalScale.get_act_zero_point()
                    if gzp is not None:
                        self.zero_point.copy_(gzp.to(self.zero_point.device))
                    return

            s, zp = compute_act_scale_minmax(
                self.running_min, self.running_max,
                self.bit_width, self.symmetric)
            self.scale.copy_(s)
            self.zero_point.copy_(zp)

    def calibrate(self, x: torch.Tensor):
        """Observe a batch during PTQ calibration."""
        self._observe(x)
        self._update_scale()
        self._calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._calibrating:
            # Calibration pass: observe stats but don't fake-quantize
            self._observe(x)
            return x
        elif self.training and not self._ptq_mode:
            # QAT: update running stats and fake-quantize
            self._observe(x)
            self._update_scale()
        elif not self._calibrated and self._ptq_mode:
            # PTQ but not yet calibrated -- pass through
            return x

        scale = self.scale
        zp = self.zero_point

        # Reshape for broadcasting
        if scale.dim() == 1 and scale.size(0) > 1 and x.dim() > 1:
            shape = [1] * x.dim()
            shape[1] = -1
            scale = scale.view(*shape)
            zp = zp.view(*shape)

        if self.symmetric:
            return fake_quantize_symmetric(x, scale, self.bit_width,
                                           self.layer_name, "activation")
        else:
            return fake_quantize_asymmetric(x, scale, zp, self.bit_width,
                                            self.layer_name, "activation")

    def extra_repr(self) -> str:
        mode = "ptq" if self._ptq_mode else "qat"
        return (f'bit_width={self.bit_width}, scale_type={self.scale_type}, '
                f'symmetric={self.symmetric}, mode={mode}')


# ============================================================================
# Membrane range check (no fake quantization, just warnings)
# ============================================================================

def check_membrane_range(mem: torch.Tensor, bit_width: int = 16,
                         mem_range: float = 2.0,
                         layer_name: str = "membrane") -> torch.Tensor:
    """
    Check if membrane potential would exceed quantization limits.
    Returns the membrane tensor UNCHANGED -- no fake quantization.
    """
    qmax = 2 ** (bit_width - 1) - 1
    qmin = -qmax - 1
    scale = mem_range / qmax

    with torch.no_grad():
        mem_scaled = mem / scale
        _warn_if_clipped(mem_scaled, qmin, qmax, layer_name, "membrane")

    return mem


# Backwards-compatible alias
def QuantMembrane(mem, bit_width=16, mem_range=2.0, layer_name="membrane"):
    return check_membrane_range(mem, bit_width, mem_range, layer_name)


# ============================================================================
# SurrogateSpike (unchanged -- needed by LIF/IF layers)
# ============================================================================

class SurrogateSpike(torch.autograd.Function):
    """Hard threshold forward; sigmoid surrogate gradient backward."""
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
        return grad_out * alpha * s * (1 - s), None


# ============================================================================
# PTQ Calibration
# ============================================================================

@torch.no_grad()
def calibrate_model(model: nn.Module, data_loader, device: str = 'cuda',
                    num_batches: int = 100, config: dict = None):
    """
    Run PTQ calibration: forward pass over data to collect activation statistics.

    After calibration, all QuantAct modules will have frozen scale/zero_point
    computed from observed min/max.

    Args:
        model: model with QuantWeight/QuantAct modules
        data_loader: calibration data loader
        device: device to run on
        num_batches: number of batches to use for calibration
        config: model config (used to get num_bins etc.)
    """
    model.eval()
    model.to(device)

    # 1. Calibrate weights (immediate -- just needs the weight tensors)
    for name, module in model.named_modules():
        if hasattr(module, 'weight_quant') and hasattr(module, 'conv'):
            module.weight_quant.calibrate(module.conv.weight)
        elif hasattr(module, 'weight_quant') and hasattr(module, 'lin'):
            module.weight_quant.calibrate(module.lin.weight)

    print("[PTQ] Weight scales calibrated.")

    # 2. Calibrate activations (needs forward passes)
    quant_act_modules = []
    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            module._calibrated = False
            module._calibrating = True
            module._num_batches_seen = 0
            quant_act_modules.append((name, module))

    print(f"[PTQ] Calibrating {len(quant_act_modules)} activation quantizers "
          f"over {num_batches} batches...")

    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break

        events = (batch['input'].to(device)
                  if isinstance(batch, dict) else batch[0].to(device))

        # Forward pass (model handles temporal bins internally with 5D input)
        _ = model(events)

        if (i + 1) % 20 == 0:
            print(f"  Calibrated {i + 1}/{num_batches} batches")

    # Mark all as calibrated and do final scale update
    for name, module in quant_act_modules:
        module._calibrating = False
        module._update_scale()
        module._calibrated = True

    # If global scale mode, compute global scales
    _compute_global_scales(model)

    print("[PTQ] Calibration complete.")
    print_scale_summary(model)


def _compute_global_scales(model: nn.Module):
    """Compute global scale as the max of all per-layer scales."""
    weight_scales = []
    act_scales = []

    for name, module in model.named_modules():
        if isinstance(module, QuantWeight) and module.scale_type != 'global':
            weight_scales.append(module.scale.max().item())
        if isinstance(module, QuantAct) and module.scale_type != 'global':
            act_scales.append(module.scale.max().item())

    if weight_scales:
        gs = torch.tensor(max(weight_scales))
        GlobalScale.set_weight_scale(gs)
    if act_scales:
        gs = torch.tensor(max(act_scales))
        GlobalScale.set_act_scale(gs)

    # Push global scales to modules that use them
    for name, module in model.named_modules():
        if isinstance(module, QuantWeight) and module.scale_type == 'global':
            s = GlobalScale.get_weight_scale()
            if s is not None:
                module.scale.copy_(s)
                module._calibrated = True
        if isinstance(module, QuantAct) and module.scale_type == 'global':
            s = GlobalScale.get_act_scale()
            if s is not None:
                module.scale.copy_(s)
                module._calibrated = True


# ============================================================================
# Export / inspection utilities
# ============================================================================

def print_scale_summary(model: nn.Module):
    """Print a summary of all quantization scales in the model."""
    print("\n" + "=" * 60)
    print("Quantization Scale Summary")
    print("=" * 60)

    for name, module in model.named_modules():
        if isinstance(module, QuantWeight):
            s = module.scale
            if s.numel() == 1:
                print(f"  {name:40s} W{module.bit_width} "
                      f"scale={s.item():.6f} ({module.scale_type})")
            else:
                print(f"  {name:40s} W{module.bit_width} "
                      f"scale=[{s.min().item():.6f}, {s.max().item():.6f}] "
                      f"({module.scale_type})")
        elif isinstance(module, QuantAct):
            s = module.scale
            if s.numel() == 1:
                print(f"  {name:40s} A{module.bit_width} "
                      f"scale={s.item():.6f} "
                      f"zp={module.zero_point.item():.1f} "
                      f"({module.scale_type})")
            else:
                print(f"  {name:40s} A{module.bit_width} "
                      f"scale=[{s.min().item():.6f}, {s.max().item():.6f}] "
                      f"({module.scale_type})")

    print("=" * 60 + "\n")


def export_quantized_params(model: nn.Module, multiplier_bits: int = 16) -> Dict[str, Dict]:
    """
    Export integer weights, scales, and requantization parameters for FPGA deployment.

    For each quantized layer, computes the requantization multiplier:

        M_real = (S_weight * S_input) / S_output

    and decomposes it into integer multiply-and-shift:

        M_real ≈ M_0 * 2^(-shift)

    so the FPGA can compute:

        out_int = (M_0 * acc_int) >> shift

    instead of floating-point multiplication.

    For SNN architectures where inter-layer signals are binary spikes (0/1),
    S_input = 1.0 for all layers (the spike value IS the integer value).

    Args:
        model: calibrated model with QuantWeight/QuantAct modules
        multiplier_bits: bit width of the integer multiplier M_0 (default 16)

    Returns dict: {layer_name: {int_weight, weight_scale, act_scale,
                                input_scale, M_real, M_0, shift, ...}}
    """
    exported = {}

    # Track the previous layer's output activation scale.
    # In an SNN, inter-layer signals are spikes (0 or 1), so S_in = 1.0.
    # For the first layer, raw sensor input is clamped to [0, 1], also S_in = 1.0.
    prev_act_scale = torch.tensor(1.0)

    for name, module in model.named_modules():
        layer_info = {}
        weight = None

        if hasattr(module, 'weight_quant') and hasattr(module, 'conv'):
            weight = module.conv.weight
        elif hasattr(module, 'weight_quant') and hasattr(module, 'lin'):
            weight = module.lin.weight

        if weight is not None:
            wq = module.weight_quant
            layer_info['int_weight'] = wq.get_int_weight(weight).cpu()
            layer_info['weight_scale'] = wq.scale.cpu()
            layer_info['weight_bit_width'] = wq.bit_width

        if hasattr(module, 'act_quant'):
            aq = module.act_quant
            layer_info['act_scale'] = aq.scale.cpu()
            layer_info['act_zero_point'] = aq.zero_point.cpu()
            layer_info['act_bit_width'] = aq.bit_width

        # Compute requantization M, M_0, shift if we have both weight and act
        if weight is not None and hasattr(module, 'act_quant'):
            wq = module.weight_quant
            aq = module.act_quant

            s_w = wq.scale.cpu().float()
            s_in = prev_act_scale.float()
            s_out = aq.scale.cpu().float()

            layer_info['input_scale'] = s_in.clone()

            # M_real = (S_w * S_in) / S_out  — per-channel if S_w is per-channel
            m_real = (s_w * s_in) / s_out.clamp(min=1e-10)

            layer_info['M_real'] = m_real

            # Decompose each element into M_0 * 2^(-shift)
            m_real_flat = m_real.view(-1)
            m0_list = []
            shift_list = []
            for val in m_real_flat:
                m0, sh = _decompose_multiplier(val.item(), multiplier_bits)
                m0_list.append(m0)
                shift_list.append(sh)

            m0_tensor = torch.tensor(m0_list, dtype=torch.int64).view(m_real.shape)
            shift_tensor = torch.tensor(shift_list, dtype=torch.int32).view(m_real.shape)

            layer_info['M_0'] = m0_tensor
            layer_info['shift'] = shift_tensor
            layer_info['multiplier_bits'] = multiplier_bits

            # Update prev_act_scale for next layer
            prev_act_scale = aq.scale.cpu()

        if layer_info:
            exported[name] = layer_info

    return exported


def _decompose_multiplier(m_real: float, m_bits: int = 16):
    """
    Decompose a real-valued multiplier M into integer M_0 and right-shift n:

        M ≈ M_0 * 2^(-n)

    This enables integer-only multiply-and-shift on FPGA:

        result = (M_0 * accumulator) >> n

    Args:
        m_real: positive real-valued multiplier
        m_bits: bit width for integer multiplier M_0

    Returns:
        (M_0, shift): M_0 is a non-negative integer fitting in m_bits,
                       shift is the right-shift amount
    """
    if m_real <= 0 or not math.isfinite(m_real):
        return 0, 0

    # Normalize m_real into [0.5, 1.0): find exponent
    exp = math.floor(math.log2(m_real))
    significand = m_real / (2.0 ** exp)  # in [1.0, 2.0)
    significand /= 2.0                   # in [0.5, 1.0)
    exp += 1

    # M_0 = round(significand * 2^m_bits)
    m0 = int(round(significand * (1 << m_bits)))
    max_val = (1 << m_bits) - 1
    m0 = min(m0, max_val)

    # M_real ≈ M_0 * 2^(-shift)  →  shift = m_bits - exp
    shift = m_bits - exp

    return m0, shift


def set_quant_mode(model: nn.Module, mode: str):
    """
    Switch all quantizers between 'ptq' and 'qat' mode.

    Args:
        model: model containing QuantWeight/QuantAct modules
        mode: 'ptq' or 'qat'
    """
    ptq = (mode == 'ptq')
    for module in model.modules():
        if isinstance(module, (QuantWeight, QuantAct)):
            module._ptq_mode = ptq
