import numpy as np

def loss_fn(spike_counts, target_class, num_classes):
    target = np.zeros(num_classes)
    target[target_class] = 1.0
    spike_counts = spike_counts / (np.sum(spike_counts) + 1e-8)
    return np.mean((spike_counts - target) ** 2)

def surrogate_grad(x, v_th, alpha=10.0):
    return alpha * np.exp(-alpha * np.abs(x - v_th)) / (1.0 + 1e-8)

def sgd_update(param, grad, lr=0.01):
    return param - lr * grad
