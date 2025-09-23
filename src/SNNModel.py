import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit, prange
from numba.typed import List

# Numba-friendly “pack” = tuple of arrays & scalars.
# Order (and dtypes) matter and must be consistent everywhere:
# (tau, v_th, neuron_id, t_last, membrane_potential_fict, membrane_potential_phys,
#  membrane_potential_map, synapse_w_matrix, bit_mask, t_bit_mask)

def build_layer_arrays(N_prev, N):
    tau         = np.full(N, 1.0, dtype=np.float32)         # Trainable
    v_th        = np.full(N, 0.9, dtype=np.float32)         # Trainable
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
def xor_reduce(ids: np.ndarray) -> np.int64:
    acc = np.int64(0)
    for i in range(ids.size):
        acc ^= ids[i]
    return acc


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

    alpha = 0.001
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
    decay = np.exp(-tau * np.float32(delta_t)).astype(np.float32)
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

    membrane_potential_phys ^= np.int64(delta_t)
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

    for i in range(membrane_potential_phys.size):
        mem_entry = membrane_potential_phys[i]
        membrane_potential_map[i, mem_entry] += alpha * (mem_entry - membrane_potential_map[i, mem_entry])


    # return only the scalar t_last and the spike indices; arrays were updated in-place
    return t_last, active_next



@njit(parallel=True, cache=True)
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
    return_intermediates: bool,
):
    num_layers = len(tau_list)
    B, T, P_, X_, Y_ = frames_batch.shape
    spike_neurons = np.full((B, max_spikes_per_sample), -1, dtype=np.int64)
    spike_times   = np.zeros((B, max_spikes_per_sample), dtype=np.int64)
    spike_counts  = np.zeros(B, dtype=np.int64)

    if return_intermediates:
        max_neurons = 0
        for arr in tau_list:
            n = arr.shape[0]
            if n > max_neurons:
                max_neurons = n
        activations = np.zeros((num_layers, B, max_neurons), dtype=np.int64)
    else:
        activations = None

    for b in prange(B):
        write_idx = 0

        # ---- per-sample local state (mutable) ----
        # Make local copies so threads don't race on shared state.
        local_t_last = List()
        local_membrane_potential_fict = List()
        local_membrane_potential_phys = List()

        for l in range(num_layers):
            local_t_last.append(np.int64(0))  
            # per-neuron arrays copied once per sample:
            local_membrane_potential_fict.append(membrane_potential_fict_list[l].copy())
            local_membrane_potential_phys.append(membrane_potential_phys_list[l].copy())

        # ---- process timesteps ----
        for t in range(T):
            frame_t = frames_batch[b, t]
            if not np.any(frame_t):
                continue

            active_p, active_x, active_y = np.where(frame_t > 0)
            active_neurons = (active_x + Y * active_y + (X * Y) * active_p).astype(np.int64)

            # push through layers; mutate only local copies
            for l in range(num_layers):
                t_last_new, active_next = layer_step(
                    active_neurons, 
                    t,
                    prev_neuron_id_list[l],
                    tau_list[l], 
                    v_th_list[l], 
                    local_t_last[l],
                    local_membrane_potential_fict[l],
                    local_membrane_potential_phys[l],
                    membrane_potential_map_list[l],
                    synapse_w_matrix_list[l],
                    m_bit_mask, 
                    t_bit_mask
                )

                local_t_last[l] = t_last_new
                active_neurons = active_next

                if active_neurons.size == 0:
                    break

                if return_intermediates:
                    activations[l][b, active_neurons] += 1


            # write final-layer spikes into preallocated buffers
            num_output_spikes = active_neurons.size
            can_write = max_spikes_per_sample - write_idx
            if num_output_spikes > 0 and can_write > 0:
                num_to_write = num_output_spikes if num_output_spikes <= can_write else can_write

                spike_neurons[b, write_idx:write_idx + num_to_write] = active_neurons[:num_to_write]

                for j in range(num_to_write):
                    spike_times[b, write_idx + j] = t
                write_idx += num_to_write

        spike_counts[b] = write_idx

    return spike_neurons, spike_times, spike_counts, activations

# ---------------------- Core driver ----------------------
def forward_batch_driver(model, frames_batch, logger=None, return_intermediates=False):
    """
    Generic forward driver. 
    - `layer_step_fn`: can be numba-compiled or pure Python (layer_step.py_func)
    - `logger`: only used in debug mode
    """
    B, T, P, X, Y = frames_batch.shape
    num_layers = len(model.layers["tau"])
    dt_dtype = model.dt_dtype
    outputs = []

    if return_intermediates:
        activations = [np.zeros((B, arr.shape[0]), dtype=np.int64) for arr in model.layers["tau"]]
    else:
        activations = None
    

    for b in range(B):
        spikes = []

        # ---- per-sample local state (mutable) ----
        # Make local copies so threads don't race on shared state.
        local_t_last = []

        for l in range(num_layers):
            local_t_last.append(np.int64(0))  

        for t in range(T):
            frame_t = frames_batch[b, t]
            if not np.any(frame_t):
                continue

            active_p, active_x, active_y = np.where(frame_t > 0)
            active_neurons = (active_x + Y * active_y + (X * Y) * active_p).astype(np.int64)

            for l in range(num_layers):
                t_last_new, active_next = layer_step.py_func(
                    active_neurons,
                    np.int64(t),
                    model.layers["prev_neuron_id"][l],
                    model.layers["tau"][l],
                    model.layers["v_th"][l],
                    local_t_last[l],  # t_last reset per timestep (simplify debug)
                    model.layers["membrane_potential_fict"][l],
                    model.layers["membrane_potential_phys"][l],
                    model.layers["membrane_potential_map"][l],
                    model.layers["synapse_w_matrix"][l],
                    model.m_bit_mask,
                    model.t_bit_mask,
                )

                local_t_last[l] = t_last_new
                active_neurons = active_next

                if active_neurons.size == 0:
                    break

                if return_intermediates:
                    activations[l][b, active_neurons] += 1

                # ----- Debug logging -----
            if logger is not None:
                logger.record_state(model, int(t))


            # collect final-layer spikes
            for n in active_neurons:
                spikes.append((int(n), int(t)))

        outputs.append(np.array(spikes, dtype=dt_dtype))

    return outputs, activations


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

            spike_neurons, spike_times, spike_counts, activations = forward_batch_numba(
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
                return_intermediates,
            )

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

    def save(self, path="../params/model_params.npz"):
        """Save all arrays for reproducibility."""
        data = {}
        for key, lst in self.layers.items():
            data[key] = [np.array(arr) for arr in lst]  # convert typed.List to numpy arrays

        np.savez(path, **data)
        print(f"✅ Model saved to {path}")

    def load(self, path="../params/model_params.npz"):
        """Restore arrays into typed lists."""
        if not os.path.exists(path):
            print(f"⚠️\t Model file {path} does not exist. Skipping load.")
            return

        npzfile = np.load(path, allow_pickle=True)
        restored = {}

        for key in self.layers.keys():
            arrs = npzfile[key]
            typed_list = List()
            for arr in arrs:
                typed_list.append(arr)
            restored[key] = typed_list

        self.layers = restored
        print(f"✅\t Model loaded from {path}")

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
