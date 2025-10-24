import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tonic
from tonic import transforms
import os

# --- local imports ---
from lif import LIFParams, LinearLIF
from SNNModel import SNNModel, SNNConfig
from trainer import SNNTrainer, SNNCriterion, TrainCfg
from analysis import *
import utils


# -----------------------------------------------------------
# Helper: build model + config
# -----------------------------------------------------------
def build_model(
    model_cfg: SNNConfig = None,
    lif_cfg: LIFParams = None,
):

    layers = nn.ModuleList(
        [
            LinearLIF(2 * 34 * 34, 1024, lif_cfg),
            nn.Dropout(0.1),
            LinearLIF(1024, 1024, lif_cfg),
            LinearLIF(1024, 10, lif_cfg),
        ]
    )

    return SNNModel(layers, model_cfg)


# -----------------------------------------------------------
# Data
# -----------------------------------------------------------
def get_dataloaders(batch_size=64, time_bins: int = 100):
    sensor_size = tonic.datasets.NMNIST.sensor_size
    transform = transforms.Compose(
        [transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_bins)]
    )
    trainset = tonic.datasets.NMNIST(
        save_to="../dataset/nmnist", train=True, transform=transform
    )
    testset = tonic.datasets.NMNIST(
        save_to="../dataset/nmnist", train=False, transform=transform
    )
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


# -----------------------------------------------------------
# Train routine
# -----------------------------------------------------------
def train(model, trainer_cfg, train_loader, test_loader, device):
    print("Testing...")
    criterion = SNNCriterion()

    trainer = SNNTrainer(model, criterion, trainer_cfg)
    trainer.fit(train_loader, test_loader, device=device)

    acc, aux = trainer.evaluate(test_loader, device)

    model.save(epoch=trainer_cfg.epochs)
    utils.save_aux(aux)
    print(f"[✔] Final test accuracy: {acc:.4f}")
    return aux


# -----------------------------------------------------------
# Eval routine
# -----------------------------------------------------------
def evaluate(model, trainer_cfg, test_loader, device):
    print("Evaluating...")
    criterion = SNNCriterion()

    trainer = SNNTrainer(model, criterion, trainer_cfg)

    acc, aux = trainer.evaluate(test_loader, device)

    print(f"[✔] Test accuracy: {acc:.4f}")

    utils.save_aux(aux)
    plot_distribution(aux["spike_counts"], title="Spike Count Distribution")
    plot_weighted_spike_distribution(model, aux["spike_counts"])
    return aux


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate SNN model")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "stats"],
        required=True,
        help="train = start new training, eval = load model and evaluate",
    )
    parser.add_argument(
        "--model_name",
        default="snn_nmnist_IF.pt",
        help="directory to save or load model",
    )
    parser.add_argument(
        "--stats_name",
        default="snn_nmnist_IF_stats.pkl",
        help="directory to save or load model",
    )
    parser.add_argument(
        "--time_bins",
        type=int,
        default=100,
        help="directory to save or load model",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    model_cfg = SNNConfig(
        T=args.time_bins,
        num_classes=10,
        model_path=os.path.join("../model", args.model_name),
    )
    lif_cfg = LIFParams(
        tau_mem=20.0,
        tau_syn=5.0,
        v_th=1.0,
        v_reset=0.0,
        refractory=0,
        learn_time_constants=False,
        surrogate_scale=5.0,
    )

    trainer_cfg = TrainCfg(
        epochs=5, lr=2e-3, weight_decay=1e-4, clip_grad=1.0, amp=True
    )

    model = build_model(model_cfg=model_cfg, lif_cfg=lif_cfg).to(device)

    if args.mode == "train":
        train_loader, test_loader = get_dataloaders()
        aux = train(model, trainer_cfg, train_loader, test_loader, device)
    elif args.mode == "eval":
        model = model.load(device=device)

        _, test_loader = get_dataloaders()
        aux = evaluate(model, trainer_cfg, test_loader, device)
    elif args.mode == "stats":
        aux = utils.load_aux()

        print(aux["membrane_traces"]["layer0"].shape)
        plot_distribution(
            aux["membrane_traces"]["layer0"], title="Membrane Potential Values"
        )
        plot_membrane_trace_with_spikes(
            aux["membrane_traces"]["layer0"][0], aux["spike_traces"]["layer0"][0], neuron_idx=0, batch_idx=0, v_th=1
        )
