import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit, prange
from numba.typed import List
from matplotlib.ticker import ScalarFormatter

# Numba-friendly “pack” = tuple of arrays & scalars.
# Order (and dtypes) matter and must be consistent everywhere:
# (tau, v_th, neuron_id, t_last, membrane_potential_fict, membrane_potential_phys,
#  membrane_potential_map, synapse_w_matrix, bit_mask, t_bit_mask)

def build_layer_arrays(N_prev, N):
    tau         = np.random.random(N).astype(np.float32)         # Trainable
    v_th        = np.random.random(N).astype(np.float32)         # Trainable
    neuron_id   = np.random.randint(0, 16, size=N, dtype=np.int64) # Trainable

    membrane_potential_fict = np.zeros(N, dtype=np.float32)
    membrane_potential_phys = np.zeros(N, dtype=np.int64)

    membrane_potential_map  = 1.0 / (1.0 + np.exp(-(np.random.random((N, 16)).astype(np.float32) * 5.0))) #Trainable
    synapse_w_matrix        = np.random.random((N_prev, N)).astype(np.float32) #Trainable

    return (tau, 
            v_th, 
            neuron_id, 
            membrane_potential_fict, 
            membrane_potential_phys,
            membrane_potential_map, 
            synapse_w_matrix,
            )


def build_layers(sizes, mem_pot_max=16):
    """
    sizes: list like [input_size, h1, h2, ..., out_size]
    returns: dict of numba.typed.Lists for each field + prev_neuron_id_list
    """
    name_list  = List() 
    num_list  = List() 

    tau_list  = List() 
    v_th_list = List()
    neuron_id_list = List()

    membrane_potential_fict_list = List()
    membrane_potential_phys_list = List()
    membrane_potential_map_list  = List()
    synapse_w_matrix_list        = List()

    prev_neuron_id_list = List()

    # synthetic input neuron IDs for layer 0 
    input_neuron_id = np.random.randint(0, mem_pot_max, size=sizes[0], dtype=np.int64)

    for li in range(1, len(sizes)):
        N_prev, N = sizes[li-1], sizes[li]

        (tau, 
         v_th, 
         neuron_id, 
         membrane_potential_fict, 
         membrane_potential_phys,
         membrane_potential_map, 
         synapse_w_matrix,
         ) = build_layer_arrays(N_prev, N)

        name_list.append(f"Layer{li-1}")
        num_list.append(N)

        tau_list.append(tau)
        v_th_list.append(v_th)
        neuron_id_list.append(neuron_id)
        membrane_potential_fict_list.append(membrane_potential_fict)
        membrane_potential_phys_list.append(membrane_potential_phys)
        membrane_potential_map_list.append(membrane_potential_map)
        synapse_w_matrix_list.append(synapse_w_matrix)

        # prev neuron ids
        if li == 1:
            prev_neuron_id_list.append(input_neuron_id)
        else:
            prev_neuron_id_list.append(neuron_id_list[li-2])  # previous layer's neuron_id

    return {
        "name": name_list,
        "num_neurons": num_list,
        "tau": tau_list,
        "v_th": v_th_list,
        "neuron_id": neuron_id_list,
        "membrane_potential_fict": membrane_potential_fict_list,
        "membrane_potential_phys": membrane_potential_phys_list,
        "membrane_potential_map": membrane_potential_map_list,
        "synapse_w_matrix": synapse_w_matrix_list,
        "prev_neuron_id": prev_neuron_id_list,
    }


@njit(cache=True)
def layer_step(
    active_prev: np.ndarray,
    t: np.int64,
    prev_neuron_id: np.ndarray,
    tau: np.ndarray,                      # float32 [N]
    v_th: np.ndarray,                     # float32 [N]
    t_last: np.int64,                     # int64 scalar
    membrane_potential_fict: np.ndarray,  # float32 [N]  (updated in-place)
    membrane_potential_phys: np.ndarray,  # int64   [N]  (updated in-place)
    membrane_potential_map: np.ndarray,   # float32 [N,16]
    synapse_w_matrix: np.ndarray,         # float32 [N_prev, N]
    m_bit_mask: np.int64,
    t_bit_mask: np.int64
):

    # ---- time update ----
    delta_t = (t - t_last) & t_bit_mask
    t_last = t

    # ---- input accumulation (ALWAYS an array) ----
    N = synapse_w_matrix.shape[1]
    weights_sum = np.zeros(N, dtype=np.float32)  # ensure array(float32, 1d)
    if active_prev.size > 0:
        # sum over selected presyn rows -> (N,)
        weights_sum += synapse_w_matrix[active_prev, :].sum(axis=0)

    # ---- fictitious decay/update (IN-PLACE) ----
    # force float32 math to avoid implicit float64
    decay = np.exp(-tau * delta_t).astype(np.float32)
    membrane_potential_fict[:] = membrane_potential_fict * decay + weights_sum

    # ---- physical XOR update (IN-PLACE) ----
    if active_prev.size > 0:
        # xor-reduce prev IDs at indices
        acc = np.int64(0)
        ids = prev_neuron_id[active_prev]
        for i in range(ids.size):
            acc ^= ids[i]
        neuron_id_in = acc
    else:
        neuron_id_in = np.int64(0)

    membrane_potential_phys ^= delta_t
    membrane_potential_phys ^= (neuron_id_in & m_bit_mask)
    membrane_potential_phys &= m_bit_mask

    # ---- spike check ----
    # first count
    cnt = 0
    for i in range(membrane_potential_phys.size):
        if membrane_potential_map[i, membrane_potential_phys[i]] >= v_th[i]:
            cnt += 1
    active_next = np.empty(cnt, dtype=np.int64)

    j = 0
    for i in range(membrane_potential_phys.size):
        if membrane_potential_map[i, membrane_potential_phys[i]] >= v_th[i]:
            active_next[j] = i
            j += 1

    # return only the scalar t_last and the spike indices; arrays were updated in-place
    return t_last, active_next



@njit(parallel=True, cache=False)
def forward_batch_numba(
    frames_batch,                 # (B, T, P, X, Y)
    X: int, Y: int, P: int,
    # per-layer arrays (lists of arrays with length = num_layers)
    tau_list,                           # Training parameter
    v_th_list,                          # Training parameter 
    membrane_potential_fict_list,       # Local state (mutable)
    membrane_potential_phys_list,       # Local state (mutable)
    membrane_potential_map_list,        # Training parameter
    synapse_w_matrix_list,              # Training parameter
    prev_neuron_id_list,                # Training parameter
    m_bit_mask: np.int64,          # 2^4 - 1
    t_bit_mask: np.int64,           # 2^4 - 1
    max_spikes_per_sample: int,
    max_neurons: int,
    return_intermediates: bool,
):

    alpha = 0.001

    num_layers = len(tau_list)
    B, T, P_, X_, Y_ = frames_batch.shape
    spike_neurons = np.full((B, max_spikes_per_sample), -1, dtype=np.int64)
    spike_times   = np.zeros((B, max_spikes_per_sample), dtype=np.int64)
    spike_counts  = np.zeros(B, dtype=np.int64)

    activations = np.zeros((num_layers, B, max_neurons), dtype=np.int64)

    delta_mem_map_batch = np.zeros((B, num_layers, max_neurons, 16), dtype=np.float32)

    local_membrane_potential_fict = np.zeros((B, num_layers, max_neurons), dtype=np.float32)
    local_membrane_potential_phys = np.zeros((B, num_layers, max_neurons), dtype=np.int64)
    local_t_last = np.zeros((B, num_layers), dtype=np.int64)

    for b in prange(B):
        write_idx = 0

        # ---- process timesteps ----
        for t in range(T):
            frame_t = frames_batch[b, t]
            if not np.any(frame_t):
                continue

            active_p, active_x, active_y = np.where(frame_t > 0)
            active_neurons = (active_x + Y * active_y + (X * Y) * active_p).astype(np.int64)

            # push through layers; mutate only local copies
            for l in range(num_layers):
                n = tau_list[l].shape[0]

                # view into the thread-local slice
                membrane_potential_fict_view = local_membrane_potential_fict[b, l, :n]
                membrane_potential_phys_view = local_membrane_potential_phys[b, l, :n]
                local_t_last_view = local_t_last[b, l]

                t_last_new, active_next = layer_step(
                    active_neurons, 
                    np.int64(t),
                    prev_neuron_id_list[l],
                    tau_list[l], 
                    v_th_list[l], 
                    local_t_last_view,
                    membrane_potential_fict_view,
                    membrane_potential_phys_view,
                    membrane_potential_map_list[l],
                    synapse_w_matrix_list[l],
                    m_bit_mask, 
                    t_bit_mask
                )

                local_t_last[b, l] = t_last_new
                active_neurons = active_next

                if active_neurons.size == 0:
                    break

                if return_intermediates:
                    for k in range(active_neurons.size):
                        idx = active_neurons[k]
                        if idx < max_neurons:
                            activations[l, b, idx] += 1

                # ---- accumulate LUT deltas in thread-local buffer (NO WRITE to shared LUT) ----
                # For all neurons i in this layer, at this time step, push:
                #   d = alpha * (bucket - mem_map[i, bucket])
                # into delta_mem_map_batch[b, l, i, bucket]
                lut = membrane_potential_map_list[l]  # read-only
                for i in range(n):
                    bucket = membrane_potential_phys_view[i]
                    # guard in case bucket out of [0,15]
                    if 0 <= bucket < 16:
                        # compute delta relative to current LUT (read-only)
                        d = alpha * (np.float32(bucket) - lut[i, bucket])
                        delta_mem_map_batch[b, l, i, bucket] += d


            # write final-layer spikes into preallocated buffers
            num_output_spikes = active_neurons.size
            can_write = max_spikes_per_sample - write_idx
            if can_write > 0:
                num_to_write = num_output_spikes if num_output_spikes <= can_write else can_write

                spike_neurons[b, write_idx:write_idx + num_to_write] = active_neurons[:num_to_write]

                for j in range(num_to_write):
                    spike_times[b, write_idx + j] = t
                write_idx += num_to_write

        spike_counts[b] = write_idx

    return spike_neurons, spike_times, spike_counts, activations, delta_mem_map_batch

# ---------------------- Core driver ----------------------
def forward_batch_driver(model, frames_batch, logger=None, return_intermediates=False):
    """
    Debug-mode forward driver. Mirrors forward_batch_numba exactly,
    but runs in pure Python and can log neuron states with `logger`.
    """
    alpha = 0.01
    B, T, P, X, Y = frames_batch.shape
    num_layers = len(model.layers["tau"])
    max_neurons = max(arr.shape[0] for arr in model.layers["tau"])
    out_size = model.layer_sizes[-1]
    max_spikes_per_sample = int(T * out_size)

    dt_dtype = model.dt_dtype
    outputs = []
    activations = [np.zeros((B, arr.shape[0]), dtype=np.int64) for arr in model.layers["tau"]] \
                  if return_intermediates else None

    # same buffers as in numba kernel
    spike_neurons = np.full((B, max_spikes_per_sample), -1, dtype=np.int64)
    spike_times   = np.zeros((B, max_spikes_per_sample), dtype=np.int64)
    spike_counts  = np.zeros(B, dtype=np.int64)
    delta_mem_map_batch = np.zeros((B, num_layers, max_neurons, 16), dtype=np.float32)

    local_membrane_potential_fict = np.zeros((B, num_layers, max_neurons), dtype=np.float32)
    local_membrane_potential_phys = np.zeros((B, num_layers, max_neurons), dtype=np.int64)
    local_t_last = np.zeros((B, num_layers), dtype=np.int64)

    for b in range(B):
        write_idx = 0
        spikes = []

        for t in range(T):
            frame_t = frames_batch[b, t]
            if not np.any(frame_t):
                continue

            active_p, active_x, active_y = np.where(frame_t > 0)
            active_neurons = (active_x + Y * active_y + (X * Y) * active_p).astype(np.int64)

            for l in range(num_layers):
                n = model.layers["tau"][l].shape[0]
                mf_view = local_membrane_potential_fict[b, l, :n]
                mp_view = local_membrane_potential_phys[b, l, :n]
                t_last_view = local_t_last[b, l]

                t_last_new, active_next = layer_step.py_func(
                    active_neurons,
                    np.int64(t),
                    model.layers["prev_neuron_id"][l],
                    model.layers["tau"][l],
                    model.layers["v_th"][l],
                    t_last_view,
                    mf_view,
                    mp_view,
                    model.layers["membrane_potential_map"][l],
                    model.layers["synapse_w_matrix"][l],
                    model.m_bit_mask,
                    model.t_bit_mask,
                )

                local_t_last[b, l] = t_last_new
                active_neurons = active_next

                if active_neurons.size == 0:
                    break

                if return_intermediates:
                    for idx in active_neurons:
                        activations[l][b, idx] += 1

                # accumulate LUT deltas (like numba kernel)
                lut = model.layers["membrane_potential_map"][l]
                for i in range(n):
                    bucket = mp_view[i]
                    if 0 <= bucket < 16:
                        d = alpha * (float(bucket) - lut[i, bucket])
                        delta_mem_map_batch[b, l, i, bucket] += d

            # optional logging
            if logger is not None:
                logger.record_state(model, int(t))

            # collect final-layer spikes
            num_output_spikes = active_neurons.size
            can_write = max_spikes_per_sample - write_idx
            if can_write > 0:
                num_to_write = min(num_output_spikes, can_write)
                spike_neurons[b, write_idx:write_idx + num_to_write] = active_neurons[:num_to_write]
                for j in range(num_to_write):
                    spike_times[b, write_idx + j] = t
                write_idx += num_to_write
                for n in active_neurons[:num_to_write]:
                    spikes.append((int(n), int(t)))

        spike_counts[b] = write_idx
        outputs.append(np.array(spikes, dtype=dt_dtype))

    # apply LUT deltas (same as numba kernel)
    delta_sum = delta_mem_map_batch.sum(axis=0)
    for l in range(num_layers):
        n = model.layers["membrane_potential_map"][l].shape[0]
        model.layers["membrane_potential_map"][l][:, :] += delta_sum[l, :n, :]

    return (outputs, activations) if return_intermediates else (outputs, None)



class model:
    def __init__(self, x=34, y=34, p=2, hidden_sizes=(256,), out_size=10):
        self.X = x
        self.Y = y
        self.P = p
        self.t_bit = 4
        self.timestep_max = 2 ** self.t_bit  # must be power of 2
        self.t_bit_mask = self.timestep_max - 1

        self.m_bit = 4
        self.mem_pot_max = 2 ** self.m_bit  # must be power of 2
        self.m_bit_mask = self.mem_pot_max - 1

        layer_sizes = [x*y*p, *hidden_sizes, out_size]
        self.layer_sizes = layer_sizes
        self.layers = build_layers(layer_sizes, mem_pot_max=self.mem_pot_max)

        self.dt_dtype = np.dtype([('n', np.int64), ('t', np.int64)])
        self.logger = None

    def __call__(self, frames_batch, debug=False, return_intermediates=False):
        if debug:
           outputs, activations = forward_batch_driver(self, frames_batch, logger=self.logger, return_intermediates=return_intermediates) 
        else:
            B, T, P, X, Y = frames_batch.shape
            out_size = self.layer_sizes[-1]

            # Safe upper bound: at most 'out_size' spikes per timestep
            max_spikes_per_sample = int(T * out_size)

            # compute max_neurons in Python (Numba-safe arg)
            max_neurons = 0
            for arr in self.layers["tau"]:
                n = arr.shape[0]
                if n > max_neurons:
                    max_neurons = n

            spike_neurons, spike_times, spike_counts, activations, delta_mem_map_batch = forward_batch_numba(
                frames_batch,
                self.X, 
                self.Y, 
                self.P,
                self.layers["tau"], 
                self.layers["v_th"], 
                self.layers["membrane_potential_fict"], 
                self.layers["membrane_potential_phys"],
                self.layers["membrane_potential_map"], 
                self.layers["synapse_w_matrix"],
                self.layers["prev_neuron_id"],
                self.m_bit_mask,
                self.t_bit_mask,
                max_spikes_per_sample,
                max_neurons,
                return_intermediates
            )

            delta_sum = delta_mem_map_batch.sum(axis=0)
            # Apply to each layer’s LUT (only the used rows per layer)
            for l in range(len(self.layers["membrane_potential_map"])):
                n = self.layers["membrane_potential_map"][l].shape[0]
                self.layers["membrane_potential_map"][l][:, :] += delta_sum[l, :n, :]

            # Pack per-sample results as structured arrays (slice by counts[b])
            outputs = []

            for b in range(B):
                batch_spike_count = int(spike_counts[b])

                if batch_spike_count == 0:
                    outputs.append(np.empty(0, dtype=self.dt_dtype))
                    continue

                out = np.empty(batch_spike_count, dtype=self.dt_dtype)
                out['n'] = spike_neurons[b, :batch_spike_count]
                out['t'] = spike_times[b, :batch_spike_count]
                outputs.append(out)

        if return_intermediates:
            return outputs, activations
        else:
            return outputs

    def save(self, path="../checkpoints/model.npz"):
        """Save all arrays for reproducibility."""
        data = {}
        for key, lst in self.layers.items():
            for i, arr in enumerate(lst):
                data[f"{key}_{i}"] = np.array(arr)  # safe conversion

        np.savez(path, **data)
        print(f"✅ Model saved to {path}")


    def load(self, path="../checkpoints/model.npz"):
        """Restore arrays into typed lists."""
        if not os.path.exists(path):
            print(f"⚠️ Model file {path} does not exist. Skipping load.")
            return

        npzfile = np.load(path)
        restored = {}

        for key in self.layers.keys():
            typed_list = List()
            # collect all arrays for this key (sorted to preserve order)
            subkeys = sorted([k for k in npzfile.files if k.startswith(key + "_")],
                             key=lambda x: int(x.split("_")[-1]))
            for subkey in subkeys:
                typed_list.append(npzfile[subkey])
            restored[key] = typed_list

        self.layers = restored
        print(f"✅ Model loaded from {path}")


    def set_logger(self, logger):
        self.logger = logger


    def __repr__(self):
        s = f"SNNModel(\n"
        s += f"  Input: {self.layer_sizes[0]}\n"
        s += f"  Hidden: {self.layer_sizes[1:-1]}\n"
        s += f"  Output: {self.layer_sizes[-1]}\n"
        s += f"  Layers:\n"
        for i, n in enumerate(self.layer_sizes):
            lname = "Input" if i == 0 else f"Layer{i-1}"
            s += f"    {lname}: {n} neurons\n"
        s += f")"
        return s

    def plot_parameters(self, param_name, bins=30, show=True, save_path=None):
        """
        Plot distribution of a parameter across all layers.
        - Discrete small-range ints → bar chart with integer ticks
        - Continuous values → histogram with bin centers labeled
        - X-axis labels auto-switch to scientific notation if too long
        """
        if param_name not in self.layers:
            raise ValueError(
                f"Parameter '{param_name}' not found in model.layers. "
                f"Available: {list(self.layers.keys())}"
            )

        arrays = [np.array(arr).ravel() for arr in self.layers[param_name]]
        values = np.concatenate(arrays)

        fig, ax = plt.subplots(figsize=(6, 4))

        # Case 1: discrete small-range ints (like neuron_id)
        if np.issubdtype(values.dtype, np.integer) and values.max() - values.min() < 64:
            unique, counts = np.unique(values, return_counts=True)
            ax.bar(unique, counts, align="center", alpha=0.7, edgecolor="black")
            ax.set_xticks(unique)
            ax.set_xlabel(param_name)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {param_name}")

        # Case 2: continuous values
        else:
            counts, edges, bars = ax.hist(values, bins=bins, edgecolor="black", alpha=0.7)
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            ax.set_xticks(bin_centers)
            ax.set_xticklabels([f"{c:.2e}" if (abs(c) >= 1e3 or abs(c) <= 1e-3) else f"{c:.2f}" 
                                for c in bin_centers], rotation=45, fontsize=8)
            ax.set_xlabel(param_name)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {param_name}")
            ax.grid(True, linestyle="--", alpha=0.6)

        # Force scientific notation globally if needed
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)


    def plot_membrane_potential_map(self, layer_idx, neuron_idx, show=True, save_path=None):
        """
        Plot the membrane potential map of a single neuron in a given layer,
        sorted by ascending value. X-axis shows the binary key values.

        Args:
            layer_idx (int): Index of the layer (0 = first hidden layer).
            neuron_idx (int): Index of the neuron in that layer.
            show (bool): Whether to display the plot immediately.
            save_path (str or None): If given, save the figure to this path.
        """
        # Extract the map row
        mem_map = np.array(self.layers["membrane_potential_map"][layer_idx][neuron_idx])

        # Generate keys (e.g. 0000..1111 for 16 buckets)
        num_keys = mem_map.shape[0]
        bit_width = int(np.ceil(np.log2(num_keys)))
        keys_bin = [format(k, f"0{bit_width}b") for k in range(num_keys)]

        # Sort values and corresponding keys
        sorted_idx = np.argsort(mem_map)
        sorted_vals = mem_map[sorted_idx]
        sorted_keys = [keys_bin[i] for i in sorted_idx]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(num_keys), sorted_vals, marker="o", linestyle="-", color="tab:blue")

        # Set binary keys as x-ticks
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