# SNNModel_Torch_Batched.py
import os
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def _xor_reduce_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    XOR-reduce along a given dim (tree reduction).
    Works for int64 tensors on CPU/GPU.
    """
    if x.size(dim) == 0:
        shape = list(x.shape)
        shape.pop(dim)
        return torch.zeros(shape, dtype=torch.int64, device=x.device)

    t = x
    while t.size(dim) > 1:
        n = t.size(dim)
        if n % 2 == 1:
            pad_shape = list(t.shape)
            pad_shape[dim] = 1
            t = torch.cat(
                [t, torch.zeros(pad_shape, dtype=t.dtype, device=t.device)], dim=dim
            )
            n += 1
        t = torch.bitwise_xor(t.narrow(dim, 0, n // 2), t.narrow(dim, n // 2, n // 2))
    return t.squeeze(dim)


class SNNLayer(nn.Module):
    """
    Batched spiking layer with XOR/parity-based physical membrane updates and
    a trainable membrane-potential LUT (mem_map).
    Maintains per-batch state: mem_fict[B, N_out], mem_phys[B, N_out], t_last[B].
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        mem_pot_bits: int = 4,  # -> mem_pot_max = 2**bits
        t_bits: int = 4,  # -> timestep modulo 2**bits
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.mem_pot_bits = mem_pot_bits
        self.t_bits = t_bits
        self.mem_pot_max = 1 << mem_pot_bits
        self.m_bit_mask = (1 << mem_pot_bits) - 1
        self.t_bit_mask = (1 << t_bits) - 1

        # ---- Trainable parameters ----
        self.tau = nn.Parameter(torch.rand(n_out, dtype=torch.float32))
        self.v_th = nn.Parameter(torch.rand(n_out, dtype=torch.float32))

        # Per-neuron static IDs (buffer)
        neuron_id_init = torch.randint(0, self.mem_pot_max, (n_out,), dtype=torch.int64)
        self.register_buffer("neuron_id", neuron_id_init)

        # Trainable membrane potential map (N_out, mem_pot_max)
        self.mem_map = nn.Parameter(
            torch.sigmoid(torch.randn(n_out, self.mem_pot_max, dtype=torch.float32))
        )

        # Trainable synapse weights (N_in, N_out)
        self.syn_w = nn.Parameter(torch.rand(n_in, n_out, dtype=torch.float32))

        # ---- Stateful buffers (will be resized to (B, *) at runtime) ----
        self.register_buffer("mem_fict", torch.zeros(1, n_out, dtype=torch.float32))
        self.register_buffer("mem_phys", torch.zeros(1, n_out, dtype=torch.int64))
        self.register_buffer("t_last", torch.zeros(1, dtype=torch.int64))

    def reset_state(self, batch_size: int):
        device = self.mem_fict.device
        self.mem_fict = torch.zeros(
            batch_size, self.n_out, dtype=torch.float32, device=device
        )
        self.mem_phys = torch.zeros(
            batch_size, self.n_out, dtype=torch.int64, device=device
        )
        self.t_last = torch.zeros(batch_size, dtype=torch.int64, device=device)

    @torch.no_grad()
    def step(
        self,
        active_mask: torch.Tensor,  # (B, N_in) boolean or 0/1 float/int
        t: int,
        prev_neuron_id: torch.Tensor,  # (N_in,) int64 IDs
        accumulate_lut_delta: Optional[torch.Tensor] = None,  # (N_out, mem_pot_max)
        alpha: float = 0.0,
    ) -> torch.Tensor:
        """
        Single-timestep update for this layer across the whole batch.
        Returns active_next_mask: (B, N_out) boolean, neurons that spiked at this step.
        Optionally accumulates per-bucket LUT deltas into 'accumulate_lut_delta'.
        """
        B = active_mask.shape[0]
        device = self.mem_fict.device

        # ---- time update ----
        delta_t = (int(t) - self.t_last) & self.t_bit_mask  # (B,)
        self.t_last[:] = int(t)

        # ---- input accumulation ----
        # weights_sum = active_rows_sum @ syn_w  -> (B, N_out)
        weights_sum = active_mask.to(torch.float32) @ self.syn_w  # (B, N_out)

        # ---- fictitious decay/update ----
        # broadcast decay: (B,1) * (1,N_out)
        decay = torch.exp(
            -self.tau.unsqueeze(0) * delta_t.unsqueeze(1).to(torch.float32)
        )
        self.mem_fict = self.mem_fict * decay + weights_sum

        # ---- physical XOR update ----
        # Mask prev IDs per batch, XOR-reduce across N_in
        masked_ids = torch.where(
            active_mask.bool(),
            prev_neuron_id.unsqueeze(0).expand(B, -1),
            torch.zeros(1, self.n_in, dtype=torch.int64, device=device),
        )
        acc = _xor_reduce_dim(masked_ids, dim=1)  # (B,)

        self.mem_phys ^= delta_t.unsqueeze(1)  # (B, N_out)
        self.mem_phys ^= acc.unsqueeze(1) & self.m_bit_mask
        self.mem_phys &= self.m_bit_mask

        # ---- spike check ----
        # v_map_vals[b, i] = mem_map[i, mem_phys[b, i]]
        mem_map_exp = self.mem_map.unsqueeze(0).expand(B, -1, -1)  # (B, N_out, K)
        v_map_vals = torch.gather(mem_map_exp, 2, self.mem_phys.unsqueeze(-1)).squeeze(
            -1
        )  # (B, N_out)
        active_next_mask = v_map_vals >= self.v_th.unsqueeze(0)  # (B, N_out)

        # ---- accumulate LUT delta (optional) ----
        if accumulate_lut_delta is not None and alpha != 0.0:
            buckets = self.mem_phys  # (B, N_out)
            current_vals = v_map_vals  # (B, N_out)
            delta = alpha * (buckets.to(torch.float32) - current_vals)  # (B, N_out)

            rows = (
                torch.arange(self.n_out, device=device).unsqueeze(0).expand(B, -1)
            )  # (B, N_out)
            flat_rows = rows.reshape(-1)  # (B*N_out,)
            flat_cols = buckets.reshape(-1)  # (B*N_out,)
            flat_vals = delta.reshape(-1)  # (B*N_out,)

            accumulate_lut_delta.index_put_(
                (flat_rows, flat_cols), flat_vals, accumulate=True
            )

        return active_next_mask


class SNNModel(nn.Module):
    """
    Torch-native SNNModel (batched) that mirrors the original Numba logic but processes
    the ENTIRE BATCH in parallel on GPU.
    Still exposes a 'layers' property compatible with SNNLogger (uses batch 0 view).
    """

    def __init__(
        self,
        x: int = 34,
        y: int = 34,
        p: int = 2,
        hidden_sizes: Tuple[int, ...] = (256,),
        out_size: int = 10,
        t_bits: int = 4,
        mem_pot_bits: int = 4,
        lut_alpha: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.X, self.Y, self.P = x, y, p
        self.input_size = x * y * p
        self.layer_sizes: List[int] = [self.input_size, *hidden_sizes, out_size]
        self.t_bits = t_bits
        self.mem_pot_bits = mem_pot_bits
        self.mem_pot_max = 1 << mem_pot_bits
        self.timestep_max = 1 << t_bits
        self.t_bit_mask = self.timestep_max - 1
        self.m_bit_mask = (1 << mem_pot_bits) - 1
        self.lut_alpha = lut_alpha

        # Build layers
        self.layers_mod = nn.ModuleList(
            [
                SNNLayer(
                    self.layer_sizes[i],
                    self.layer_sizes[i + 1],
                    mem_pot_bits=mem_pot_bits,
                    t_bits=t_bits,
                )
                for i in range(len(self.layer_sizes) - 1)
            ]
        )

        # Input pseudo-neuron IDs (buffer)
        input_ids = torch.arange(self.input_size, dtype=torch.int64)
        self.register_buffer("input_neuron_id", input_ids)

        # Optional logger
        self.logger = None

        # Device handling
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Structured dtype for output events (n, t)
        self.dt_dtype = np.dtype([("n", np.int64), ("t", np.int64)])

    # -------- Public API --------
    def set_logger(self, logger):
        self.logger = logger

    def _reset_state_batched(self, B: int):
        for layer in self.layers_mod:
            layer.reset_state(B)

    @torch.no_grad()
    def forward(
        self,
        frames_batch: Any,  # np.ndarray or torch.Tensor, shape (B, T, P, X, Y)
        return_intermediates: bool = False,
        debug: bool = False,
    ) -> Any:
        """
        Run batch inference with the exact same spike propagation logic as the old model,
        but processing the entire batch (B) in parallel each timestep.
        Returns list (len B) of structured arrays [('n','t')], like before.
        """
        device = self.device

        # Accept numpy input for drop-in compatibility
        if isinstance(frames_batch, np.ndarray):
            frames_batch = torch.from_numpy(frames_batch)

        frames_batch = frames_batch.to(device)
        B, T, P, X, Y = frames_batch.shape

        # Initialize per-batch state
        self._reset_state_batched(B)

        # For compatibility: collect activations (B, N) per layer if requested
        if return_intermediates:
            activations = [
                torch.zeros(B, layer.n_out, dtype=torch.int64, device=device)
                for layer in self.layers_mod
            ]
        else:
            activations = None

        # Accumulator for LUT deltas per layer
        delta_mem_map_sum: List[torch.Tensor] = [
            torch.zeros(
                layer.n_out, layer.mem_pot_max, dtype=torch.float32, device=device
            )
            for layer in self.layers_mod
        ]

        outputs_spike_lists: List[List[Tuple[int, int]]] = [[] for _ in range(B)]

        for t in range(T):
            frame_t = frames_batch[:, t]  # (B, P, X, Y)
            if not torch.any(frame_t > 0):
                continue

            # Build (B, N_in) boolean mask of active presyn neurons
            active_mask_in = (frame_t > 0).reshape(B, -1)  # (B, P*X*Y)

            prev_ids = self.input_neuron_id
            active_mask = active_mask_in
            for li, layer in enumerate(self.layers_mod):
                active_mask = layer.step(
                    active_mask=active_mask,
                    t=int(t),
                    prev_neuron_id=prev_ids,
                    accumulate_lut_delta=delta_mem_map_sum[li],
                    alpha=self.lut_alpha,
                )
                if return_intermediates:
                    activations[li] += active_mask.to(torch.int64)
                prev_ids = layer.neuron_id

            # Final-layer spikes: active_mask (B, N_out) True -> append (n, t) per sample
            b_idx, n_idx = torch.nonzero(active_mask, as_tuple=True)
            if b_idx.numel() > 0:
                # convert to per-sample lists
                for bb, nn_idx in zip(b_idx.tolist(), n_idx.tolist()):
                    outputs_spike_lists[bb].append((int(nn_idx), int(t)))

            if self.logger is not None and debug:
                self.logger.record_state(self, int(t))

        # Convert to structured numpy arrays
        outputs: List[np.ndarray] = []
        for b in range(B):
            spikes = outputs_spike_lists[b]
            if len(spikes) == 0:
                outputs.append(np.empty(0, dtype=self.dt_dtype))
            else:
                arr = np.empty(len(spikes), dtype=self.dt_dtype)
                arr["n"] = [n for (n, _) in spikes]
                arr["t"] = [tt for (_, tt) in spikes]
                outputs.append(arr)

        # Apply accumulated LUT deltas after processing the whole batch
        for li, layer in enumerate(self.layers_mod):
            layer.mem_map.data.add_(delta_mem_map_sum[li])

        if return_intermediates:
            acts_np = [a.detach().cpu().numpy() for a in activations]
            return outputs, acts_np
        else:
            return outputs

    # -------- Save/Load --------
    def save(self, path: str = "../checkpoints/model_torch_batched.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"✅ Torch batched model saved to {path}")

    def load(
        self,
        path: str = "../checkpoints/model_torch_batched.pt",
        map_location: Optional[str] = None,
    ):
        if not os.path.exists(path):
            print(f"⚠️ Model file {path} does not exist. Skipping load.")
            return
        self.load_state_dict(torch.load(path, map_location=map_location))
        print(f"✅ Torch batched model loaded from {path}")

    # -------- Logger compatibility view (batch 0 snapshot) --------
    @property
    def layers(self) -> Dict[str, List[np.ndarray]]:
        """
        Property that mimics the old nn.layers structure so SNNLogger continues to work.
        Since states are batched, we expose the **first batch element** for logger inspection.
        """
        names = [f"Layer{i}" for i in range(len(self.layers_mod))]
        num_neurons = [layer.n_out for layer in self.layers_mod]

        tau = [layer.tau.detach().cpu().numpy() for layer in self.layers_mod]
        v_th = [layer.v_th.detach().cpu().numpy() for layer in self.layers_mod]
        neuron_id = [
            layer.neuron_id.detach().cpu().numpy() for layer in self.layers_mod
        ]

        # Take batch 0 for stateful buffers
        mem_fict = [
            layer.mem_fict.detach().cpu().numpy()[0] for layer in self.layers_mod
        ]
        mem_phys = [
            layer.mem_phys.detach().cpu().numpy()[0] for layer in self.layers_mod
        ]
        mem_map = [layer.mem_map.detach().cpu().numpy() for layer in self.layers_mod]
        syn_w = [layer.syn_w.detach().cpu().numpy() for layer in self.layers_mod]

        prev_neuron_id = []
        for i, layer in enumerate(self.layers_mod):
            if i == 0:
                prev_neuron_id.append(self.input_neuron_id.detach().cpu().numpy())
            else:
                prev_neuron_id.append(
                    self.layers_mod[i - 1].neuron_id.detach().cpu().numpy()
                )

        return {
            "name": names,
            "num_neurons": num_neurons,
            "tau": tau,
            "v_th": v_th,
            "neuron_id": neuron_id,
            "membrane_potential_fict": mem_fict,
            "membrane_potential_phys": mem_phys,
            "membrane_potential_map": mem_map,
            "synapse_w_matrix": syn_w,
            "prev_neuron_id": prev_neuron_id,
        }

    # -------- Plot helpers (compatible with old API) --------
    def plot_parameters(
        self,
        param_name: str,
        bins: int = 30,
        show: bool = True,
        save_path: Optional[str] = None,
    ):
        layers = self.layers
        if param_name not in layers:
            raise ValueError(
                f"Parameter '{param_name}' not found in model.layers. "
                f"Available: {list(layers.keys())}"
            )

        arrays = [np.array(arr).ravel() for arr in layers[param_name]]
        arrays = [a for a in arrays if a.size > 0 and np.issubdtype(a.dtype, np.number)]
        if len(arrays) == 0:
            print(f"⚠️ Nothing numeric to plot for '{param_name}'.")
            return

        values = np.concatenate(arrays)

        fig, ax = plt.subplots(figsize=(6, 4))

        if np.issubdtype(values.dtype, np.integer) and values.max() - values.min() < 64:
            unique, counts = np.unique(values, return_counts=True)
            ax.bar(unique, counts, align="center", alpha=0.7, edgecolor="black")
            ax.set_xticks(unique)
            ax.set_xlabel(param_name)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {param_name}")
        else:
            counts, edges, bars = ax.hist(
                values, bins=bins, edgecolor="black", alpha=0.7
            )
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            ax.set_xticks(bin_centers)
            ax.set_xticklabels(
                [
                    f"{c:.2e}" if (abs(c) >= 1e3 or abs(c) <= 1e-3) else f"{c:.2f}"
                    for c in bin_centers
                ],
                rotation=45,
                fontsize=8,
            )
            ax.set_xlabel(param_name)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {param_name}")
            ax.grid(True, linestyle="--", alpha=0.6)

        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_membrane_potential_map(
        self,
        layer_idx: int,
        neuron_idx: int,
        show: bool = True,
        save_path: Optional[str] = None,
    ):
        layer = self.layers_mod[layer_idx]
        mem_map = layer.mem_map.detach().cpu().numpy()[neuron_idx]

        num_keys = mem_map.shape[0]
        bit_width = int(np.ceil(np.log2(num_keys)))
        keys_bin = [format(k, f"0{bit_width}b") for k in range(num_keys)]

        sorted_idx = np.argsort(mem_map)
        sorted_vals = mem_map[sorted_idx]
        sorted_keys = [keys_bin[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            range(num_keys), sorted_vals, marker="o", linestyle="-", color="tab:blue"
        )
        ax.set_xticks(range(num_keys))
        ax.set_xticklabels(sorted_keys, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Membrane Potential Map - Layer {layer_idx}, Neuron {neuron_idx}")
        ax.set_xlabel("Key (binary)")
        ax.set_ylabel("Membrane potential value")
        ax.grid(True, linestyle="--", alpha=0.6)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # -------- Pretty repr --------
    def __repr__(self):
        s = "SNNModel_Torch_Batched(\n"
        s += f"  Input: {self.layer_sizes[0]}\n"
        s += f"  Hidden: {self.layer_sizes[1:-1]}\n"
        s += f"  Output: {self.layer_sizes[-1]}\n"
        s += f"  Layers:\n"
        for i, n in enumerate(self.layer_sizes):
            lname = "Input" if i == 0 else f"Layer{i-1}"
            s += f"    {lname}: {n} neurons\n"
        s += ")"
        return s
