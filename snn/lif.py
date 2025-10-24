import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from dataclasses import dataclass
from typing import List

class FastSigmoidSpike(Function):
    @staticmethod
    def forward(ctx, v, scale=5.0):
        ctx.save_for_backward(v)
        ctx.scale = scale
        return (v > 0).to(v.dtype)
    @staticmethod
    def backward(ctx, grad_output):
        v, = ctx.saved_tensors
        scale = ctx.scale
        grad = 1.0 / (scale * v.abs() + 1.0) ** 2
        return grad_output * grad, None

spike_fn = FastSigmoidSpike.apply

@dataclass
class LIFParams:
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    v_th: float = 1.0
    v_reset: float = 0.0
    refractory: int = 0
    learn_time_constants: bool = False
    surrogate_scale: float = 5.0
    use_IF: bool = True

class LIFCell(nn.Module):
    def __init__(self, size, cfg: LIFParams):
        super().__init__()
        self.cfg = cfg
        self.size = size
        if cfg.learn_time_constants:
            self._tau_mem = nn.Parameter(torch.tensor(cfg.tau_mem))
        else:
            self.register_buffer("_tau_mem", torch.tensor(cfg.tau_mem))

        self.register_buffer("v_th", torch.tensor(cfg.v_th))
        self.register_buffer("v_reset", torch.tensor(cfg.v_reset))
        self.scale = cfg.surrogate_scale

    def alpha(self):
        if not self.cfg.use_IF:
            if isinstance(self._tau_mem, nn.Parameter):
                tau_m = F.softplus(self._tau_mem) + 1e-3
            else:
                tau_m = self._tau_mem
            return torch.exp(-1.0 / tau_m)
        else:
            return torch.tensor(1)
            

    def init_state(self, batch, device):
        v = torch.zeros((batch, *self.size), device=device)
        return v

    def forward(self, syn, state):
        v = state
        alpha_m = self.alpha()

        v = alpha_m * v + syn
        s = spike_fn(v - self.v_th, self.scale)

        # v = v - s * (self.v_th - self.v_reset) # Soft reset
        v_next = torch.clamp(v - s * v, 0.0, 1.0) # Hard reset

        # Return v (before reset) for plotting
        return s, v, v_next

class LinearLIF(nn.Module):
    def __init__(self, in_features, out_features, cfg: LIFParams):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(in_features, out_features)
        self.cell = LIFCell((out_features,), cfg)

    def init_state(self, batch, device):
        return self.cell.init_state(batch, device)

    def forward(self, x_t, state):
        if not torch.is_floating_point(x_t):
            x_t = x_t.float()

        syn = self.fc(x_t)

        s, v, v_next = self.cell(syn, state)
        return s, v, v_next