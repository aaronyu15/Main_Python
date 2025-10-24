import os
import torch
import random
import pickle


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_of(x: torch.Tensor):
    return x.device


def save_aux(
    aux=None,
    stats_path: os.path = os.path.join("../model", "snn_nmnist_IF_stats.pkl")
):
    with open(stats_path, "wb") as f:
        pickle.dump(aux, f)

    if aux is not None:
        print(f"[✔] Saved stats → {stats_path}")


def load_aux(stats_path: os.path = os.path.join("../model", "snn_nmnist_IF_stats.pkl")):
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            aux = pickle.load(f)
        print(f"[✔] Loaded stats from {stats_path}")
        return aux
