import numpy as np

def loss_fn(spike_counts, target_class, num_classes):
    """
    Mean squared error between normalized spike counts and one-hot label.
    """
    target = np.zeros(num_classes)
    target[target_class] = 1.0
    spike_counts = spike_counts / (np.sum(spike_counts) + 1e-8)
    return np.mean((spike_counts - target) ** 2), (spike_counts - target)

def surrogate_grad(x, v_th, alpha=10.0):
    """
    Smooth approximation of derivative of step function.
    Returns gradient wrt input potential.
    """
    return alpha * np.exp(-alpha * np.abs(x - v_th)) / (1.0 + 1e-8)

def sgd_update(param, grad, lr=0.01):
    """
    Simple SGD update rule.
    """
    return param - lr * grad

def update_output_layer(layer, error_grad, lr=0.01):
    """
    Update the output layer membrane potential map using surrogate gradients.

    layer: Neuron_L instance (output layer)
    error_grad: vector of shape (num_neurons,), per-neuron error signal
    """
    grads = np.zeros_like(layer.membrane_potential_map)

    for i in range(layer.num_neurons):
        phys_val = layer.membrane_potential_phys[i]
        spike_val = layer.membrane_potential_map[i, phys_val]

        grad = error_grad[i % len(error_grad)] * surrogate_grad(spike_val, layer.v_th[i])
        grads[i, phys_val] = grad

    # Apply SGD
    layer.membrane_potential_map = sgd_update(layer.membrane_potential_map, grads, lr)

def backprop_error(next_layer, curr_layer, next_error):
    """
    Compute error for current layer given next layer errors and synapse weights.
    next_layer : Neuron_L (already updated with errors)
    curr_layer : Neuron_L (previous hidden layer)
    next_error : np.array shape (next_layer.num_neurons,) – error signals
    """
    # Backpropagate error via synapse weights
    W = next_layer.synapse.synapse_w_matrix  # shape (curr_layer.num_neurons, next_layer.num_neurons)
    return W @ next_error  # error signal for current layer neurons


def compute_layer_grads(layer, error_signal):
    grads = np.zeros_like(layer.membrane_potential_map)

    for i in range(layer.num_neurons):
        phys_val = layer.membrane_potential_phys[i]
        spike_val = layer.membrane_potential_map[i, phys_val]

        grad = error_signal[i] * surrogate_grad(spike_val, layer.v_th[i])
        grads[i, phys_val] = grad

    return grads
