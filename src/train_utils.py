import numpy as np

def batch_loss_fn(spike_counts_batch, target_classes):
    """
    Cross-entropy loss for a batch.
    
    spike_counts_batch: (B, num_classes)
    target_classes: (B,) array of class indices
    num_classes: int
    
    Returns:
        losses: (B,) loss per sample
        grads: (B, num_classes) gradient wrt spike counts
    """
    B = spike_counts_batch.shape[0]
    sums = np.sum(spike_counts_batch, axis=1, keepdims=True) + 1e-8
    probs = spike_counts_batch / sums

    # Build one-hot targets
    targets = np.zeros_like(probs)
    targets[np.arange(B), target_classes] = 1.0

    # Cross-entropy loss per sample
    losses = -np.log(probs[np.arange(B), target_classes] + 1e-8)

    # Gradient wrt spike counts
    grads = (probs - targets) / (spike_counts_batch + 1e-8)

    return losses, grads

def surrogate_grad(x, v_th, alpha=10.0):
    """
    Smooth approximation of derivative of step function.
    Returns gradient wrt input potential.
    """
    return alpha * np.exp(-alpha * np.abs(x - v_th)) / (1.0 + 1e-8)


def backprop_error_batch(layers, l, next_error_batch):
    """
    Batched backpropagation of error signals.
    
    layers: nn.layers dict
    l: current layer index
    next_error_batch: (B, N_curr) error signals
    
    Returns:
        error_batch: (B, N_prev)
    """
    W = layers["synapse_w_matrix"][l]  # shape (N_prev, N_curr)
    # Matrix multiply each batch: (B, N_curr) @ (N_curr, N_prev) -> (B, N_prev)
    return next_error_batch @ W.T



def compute_layer_grads_batch(grads, nn, l, error_batch, batch_activations):
    """
    Compute gradients for one layer l, across the whole batch.
    
    grads: dict of lists
    nn: the model
    l: layer index
    error_batch: (B, N_curr) error signals for this layer
    batch_activations: (B, N_prev) presynaptic activations for this layer
    """
    tau       = nn.layers["tau"][l]
    v_th      = nn.layers["v_th"][l]
    neuron_id = nn.layers["neuron_id"][l]
    mem_map   = nn.layers["membrane_potential_map"][l]
    syn_w     = nn.layers["synapse_w_matrix"][l]
    mem_phys  = nn.layers["membrane_potential_phys"][l]

    B, N_curr = error_batch.shape
    N_prev = batch_activations.shape[1]

    # ---- gradients wrt synapse weights ----
    # dL/dW = X^T @ error, shape (N_prev, N_curr)
    grads["synapse_w_matrix"][l] += batch_activations.T @ error_batch
    grads["tau"][l] += error_batch.sum(axis=0)
    grads["v_th"][l] += error_batch.sum(axis=0)


    return grads


def update_layer_params(grads, nn, lr=0.01, batch_size=1):
    """
    Apply accumulated gradients to update the network parameters.
    
    grads: dict of lists (same structure as nn.layers)
    nn:    model (with nn.layers dict of typed.Lists)
    lr:    learning rate
    batch_size: normalize gradients over this batch
    """

    lr = {
        "synapse_w_matrix": 0.01,
        "tau": 0.001,
        "v_th": 0.01,
    }

    num_layers = len(nn.layers["tau"])

    for l in range(num_layers):
        # Update synapse weights
        nn.layers["synapse_w_matrix"][l] -= (
            lr["synapse_w_matrix"] * grads["synapse_w_matrix"][l] / batch_size
        ).astype(np.float32)

        # Update tau
        nn.layers["tau"][l] += (
            lr["tau"] * grads["tau"][l] / batch_size
        ).astype(np.float32)

        # Update v_th
        nn.layers["v_th"][l] += (
            lr["v_th"] * grads["v_th"][l] / batch_size
        ).astype(np.float32)

        nn.layers["tau"][l] = np.clip(nn.layers["tau"][l], 1e-3, 10.0)
        nn.layers["v_th"][l] = np.clip(nn.layers["v_th"][l], 0.1, 10.0)




def init_grads(nn):
    grads = {}
    for key in ["tau", "v_th", "neuron_id", "membrane_potential_map", "synapse_w_matrix"]:
        grads[key] = [np.zeros_like(arr) for arr in nn.layers[key]]
    return grads
