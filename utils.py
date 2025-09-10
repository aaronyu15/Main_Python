import numpy as np
import matplotlib.pyplot as plt

def plot_raster(events, title="Raster Plot", figsize=(10, 6), marker_size=2):
    """
    Plots a raster plot of spikes.
    
    Parameters
    ----------
    events : np.ndarray
        Structured array or recarray with fields 'n' (neuron ID) and 't' (time).
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    marker_size : int
        Size of the dots in the raster plot.
    """
    if not isinstance(events, np.ndarray):
        raise ValueError("events must be a numpy array")
    if 'n' not in events.dtype.names or 't' not in events.dtype.names:
        raise ValueError("events array must have fields 'n' and 't'")
    
    neuron_ids = events['n']
    times = events['t']
    
    plt.figure(figsize=figsize)
    plt.scatter(times, neuron_ids, s=marker_size, color='black')
    plt.xlabel("Time")
    plt.ylabel("Neuron ID")
    plt.title(title)
    plt.tight_layout()
    plt.show()

