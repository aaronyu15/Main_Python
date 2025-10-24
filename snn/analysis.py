import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_distribution(data_dict, bins=100, title="", stat="percent"):
    vals = []
    if( isinstance(data_dict, dict)):
        for name, T in data_dict.items():
            if isinstance(T, torch.Tensor):
                T = T.detach().cpu().numpy()
            vals.append(T.reshape(-1))
        vals=np.concatenate(vals)
    else:
        if isinstance(data_dict, torch.Tensor):
            T = data_dict.detach().cpu().numpy()
        vals = T.reshape(-1)

    sns.histplot(vals, bins=bins, color="black", stat=stat)
    plt.title(title or "Distribution")
    plt.xlabel("Value")
    plt.ylabel("% of total" if stat == "percent" else stat)
    plt.grid(alpha=0.3)
    plt.show()


def plot_weighted_spike_distribution(model, spike_counts, bins=100):
    vals = []
    for name, layer in model.named_modules():
        if hasattr(layer, "fc"):
            w = layer.fc.weight.detach().cpu().abs()
            s = spike_counts.get(name, None)
            if s is None:
                continue
            s = s.detach().cpu()
            if s.numel() == w.shape[1]:
                weighted = w * s.unsqueeze(0)
            elif s.numel() == w.shape[0]:
                weighted = w * s.unsqueeze(1)
            else:
                continue
            vals.append(weighted.flatten().numpy())
    if not vals:
        return
    vals = np.concatenate(vals)
    sns.histplot(vals, bins=bins, color="black", stat="percent")
    plt.title("Weighted Spike Distribution")
    plt.xlabel("|w| × spikes")
    plt.ylabel("% of total")
    plt.show()

def plot_membrane_trace_with_spikes(
    v_trace,
    s_trace,
    neuron_idx=0,
    batch_idx=0,
    v_th=None,
    title=None,
    save_path=None
):
    """
    Plot membrane potential and spike activity of a neuron over time.

    Args:
        v_trace (Tensor): membrane potential trace (T, B, N)
        s_trace (Tensor): spike trace (T, B, N), 0/1 spikes
        neuron_idx (int): neuron index to visualize
        batch_idx (int): batch index
        v_th (float): optional spike threshold line
        title (str): optional plot title
        save_path (str): optional file path to save figure
    """
    if isinstance(v_trace, torch.Tensor):
        v_trace = v_trace.detach().cpu()
    if isinstance(s_trace, torch.Tensor):
        s_trace = s_trace.detach().cpu()

    T = v_trace.shape[0]
    v_t = v_trace[:, batch_idx, neuron_idx]
    s_t = s_trace[:, batch_idx, neuron_idx]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # --- Membrane potential ---
    ax1.plot(range(T), v_t, color="black", lw=1.5)
    if v_th is not None:
        ax1.axhline(v_th, color="red", ls="--", label="threshold")
        ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylabel("Membrane potential (V)")
    ax1.set_title(title or f"Neuron {neuron_idx} membrane and spike trace")
    ax1.grid(alpha=0.3)

    # --- Spikes ---
    ax2.bar(range(T), s_t, color="gray", width=0.8)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Spike")
    ax2.set_yticks([0, 1])
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[✔] Saved → {save_path}")
    plt.show()