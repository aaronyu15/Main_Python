import torch
import torch.nn as nn
import os
import pickle
from lif import LinearLIF
from dataclasses import dataclass, field
from typing import List


@dataclass
class SNNConfig:
    T : int = 100
    num_classes :int = 10
    model_path : os.path = os.path.join("../model", "snn_nmnist_IF.pt")
    record_v : bool = True
    record_neuron: List[int] = field(default_factory=lambda: [0])
    record_layers: List[str] = field(default_factory=lambda: ["layer0"])


class SNNModel(nn.Module):
    def __init__(self, layers: nn.ModuleList, cfg):
        super().__init__()
        self.layers = layers
        self.cfg = cfg

    # --------------------------
    #  Forward
    # --------------------------
    def forward(self, x):
        B, T, C, H, W = x.shape # (B, T, C, H, W)
        x = x.permute(1, 0, 2, 3, 4)  # → (T, B, C, H, W)

        x = x.float()
        x = x / (x.max() + 1e-6)

        x = x.reshape(T, B, -1)

        record_v = self.cfg.record_v
        states = {}
        spike_counts = {}
        membrane_traces = {} if record_v else None
        spike_traces = {} if record_v else None

        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLIF):

                states[f"layer{i}"] = layer.init_state(B, x.device)

                spike_counts[f"layer{i}"] = torch.zeros(
                    (B, layer.fc.out_features), device=x.device
                )

                if record_v:
                    if f"layer{i}" in self.cfg.record_layers:
                        membrane_traces[f"layer{i}"] = []
                        spike_traces[f"layer{i}"] = []

        for t in range(T):
            out = x[t]

            for i, layer in enumerate(self.layers):
                if not isinstance(layer, LinearLIF):
                    out = layer(out)
                    continue
                
                s, v, v_next = layer(out, states[f"layer{i}"])

                spike_counts[f"layer{i}"] += s

                if record_v:
                    if f"layer{i}" in self.cfg.record_layers:
                        membrane_traces[f"layer{i}"].append(v[:, self.cfg.record_neuron].detach().cpu())
                        spike_traces[f"layer{i}"].append(s[:, self.cfg.record_neuron].detach().cpu())

                states[f"layer{i}"] = v_next
                out = s

        if record_v:
            for k in membrane_traces.keys():
                membrane_traces[k] = torch.stack(membrane_traces[k], dim=0)
                spike_traces[k] = torch.stack(spike_traces[k], dim=0)

        last_key = list(spike_counts.keys())[-1]
        logits = spike_counts[last_key]
        
        return logits, {
            "spike_counts": spike_counts,
            "membrane_traces": membrane_traces if record_v else None,
            "spike_traces": spike_traces if record_v else None,
        }

    # --------------------------
    #  Save / Load
    # --------------------------
    def save(self, epoch=None):
        """Save parameters to model.pt and stats to stats.pkl"""
        model_path = self.cfg.model_path

        dir_path = os.path.dirname(model_path)
        os.makedirs(dir_path, exist_ok=True)

        torch.save(
            {"state_dict": self.state_dict(), "cfg": self.cfg, "epoch": epoch},
            model_path,
        )
        print(f"[✔] Saved model → {model_path}")


    def load(self, device=None):
        """Load model and optional stats."""
        model_path = self.cfg.model_path
        ckpt = torch.load(model_path, map_location=device, weights_only=False)

        self.load_state_dict(ckpt["state_dict"])
        self.to(device)
        print(f"[✔] Loaded model from {model_path}")

        return self
