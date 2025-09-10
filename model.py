import random
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt


class Base_L:
    def __init__(self, num_neurons, layer_str=""):
        self.neuron_id = np.random.randint(0, 16, size=num_neurons)
        self.num_neurons = num_neurons 
        self.layer_str = layer_str
        self.synapse = None

class Neuron_L(Base_L):
    def __init__(self, num_neurons, alpha=0.01, mode="linear", layer_str=""):
        super().__init__(self, num_neurons, layer_str=layer_str)
        self.prev_layer = None  # previous layer reference

        # Per-neuron parameters
        self.tau = np.ones(num_neurons)                   # each neuron can learn its own tau
        self.v_th = np.full(num_neurons, 0.8)             # thresholds
        self.neuron_id = np.random.randint(0, 16, size=num_neurons)

        # Per-neuron states
        self.t_last = np.zeros(num_neurons, dtype=np.int64)
        self.membrane_potential_fict = np.zeros(num_neurons)
        self.membrane_potential_phys = np.zeros(num_neurons, dtype=np.int64)

        # Each neuron has its own membrane potential map (trainable)
        self.membrane_potential_map = 1 / (1 + np.exp(-(np.random.rand(num_neurons, 16) * 5)))

        # Masks
        self.bit_len = 4
        self.bit_mask = 2**self.bit_len - 1
        self.t_bit_len = 4
        self.t_bit_mask = 2**self.t_bit_len - 1

        self.idx = np.arange(num_neurons)

    def __call__(self, n_num, t):
        if isinstance(n_num, (int, np.int64)):
            n_num = [n_num]

        if len(n_num) == 0:
            return []

        # Batch weight updates
        weights_sum = np.sum(self.synapse.synapse_w_matrix[n_num], axis=0)

        # Presynaptic IDs (XOR all together to avoid loop)
        neuron_id_in = np.bitwise_xor.reduce(self.prev_layer.neuron_id[n_num])

        # Time decay
        delta_t = (t - self.t_last) & self.t_bit_mask
        self.t_last[:] = t

        decay_factor = np.exp(-self.tau * delta_t)
        self.membrane_potential_fict = self.membrane_potential_fict * decay_factor + weights_sum

        # Physical potential update
        self.membrane_potential_phys ^= delta_t
        self.membrane_potential_phys ^= (neuron_id_in & self.bit_mask)
        self.membrane_potential_phys &= self.bit_mask

        # Spike check
        spike_vals = self.membrane_potential_map[self.idx, self.membrane_potential_phys]
        spiking_neurons = np.where(spike_vals >= self.v_th)[0]

        return spiking_neurons.tolist()


class synapse_set:
    def __init__(self, first_layer, second_layer):
        self.first_layer_num_neurons = first_layer.num_neurons
        self.second_layer_num_neurons = second_layer.num_neurons

        # For now, y = wx
        self.synapse_w_matrix = np.random.rand(self.first_layer_num_neurons, self.second_layer_num_neurons) 

        self.params = {
            "synapse_w_matrix" : self.synapse_w_matrix
        }
        
class model:
    def __init__(self, layers=[]):
        """
        Assume that the input layer fully connects to the first hidden layer by default.
        Input layer - Synapse_set - Hidden Layer 1 - Synapse_set - Hidden Layer 2 - ... - Synapse_set - Output Layer 
        Number of synapse sets = Number of layers (not accounting for input layer)
        
        """
        self.layer_seq = layers
        self.num_layers = len(self.layer_seq)
        self.num_neurons = sum([layer.num_neurons for layer in self.layer_seq])

        self.frame_x_size = 34
        self.frame_y_size = 34

        self.max_timestep = 48000
        self.num_timesteps = 16
        self.timestep = self.max_timestep // self.num_timesteps


        self.synapse_seq = [synapse_set(self.layer_seq[x], self.layer_seq[x+1]) for x in range(self.num_layers -1)]

        self.register_synapses()

        self.dt = np.dtype([
            ('n', np.int64),
            ('t', np.int64),
        ])


    def __call__(self, x):
        # Input is array of x, y, t, p values
        t = 0
        out_spikes = []

        # Event based data, no frames!
        for event in tqdm(x, desc="Processing events"):
            # Convert the 34x34 location to a neuron index
            n_num = event['x'] + self.frame_y_size * event['y']
            t = event['t']
            
            timestep_n = t % self.timestep
            for layer in self.layer_seq[1:]:
                out = layer(n_num, timestep_n)
                n_num = out

            if len(out) == 0:
                break

            for out_n in out:
                out_spikes.append((out_n, t))

        out_spikes = np.array(out_spikes, dtype=self.dt)
        return out_spikes

    def register_synapses(self):
        for i in range(self.num_layers - 1):
            self.layer_seq[i + 1].synapse = self.synapse_seq[i]
            self.layer_seq[i + 1].prev_layer = self.layer_seq[i]
            
    def backward(self, target_y, predict_y):
        for i, layer in enumerate(self.layers):
            layer.backward(target_y=target_y[i], predict_y=predict_y[i])