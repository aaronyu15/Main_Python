# model_train.py

import numpy as np
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import model, Neuron_L, Base_L  # import your model definition
from NMNIST_Dataset import NMNIST_Dataset  # custom dataset wrapper

def collate_numpy(batch):
    # batch is list of (events, label)
    events = [b[0] for b in batch]  # keep as list of numpy arrays
    labels = [b[1] for b in batch]
    return events, np.array(labels)

def load_nmnist(batch_size=1, samples=100, classes={0,1,2}):
    """
    Load NMNIST dataset with tonic.
    Converts events into numpy arrays with fields x, y, t, p.
    """

    train_dataset = tonic.datasets.NMNIST(save_to="./dataset/nmnist", train=True , transform=None)
    test_dataset  = tonic.datasets.NMNIST(save_to="./dataset/nmnist", train=False, transform=None)

    # Filter dataset to only include specified classes and limit samples
    train_dataset = NMNIST_Dataset(train_dataset, allowed_classes=classes, max_samples=samples)
    test_dataset  = NMNIST_Dataset(test_dataset , allowed_classes=classes, max_samples=samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True , collate_fn=collate_numpy)
    test_loader  = DataLoader(test_dataset , batch_size=batch_size, shuffle=False, collate_fn=collate_numpy)

    return train_loader, test_loader



def train_one_epoch(nn_model, train_loader, optimizer=None):
    """
    Simple training loop — teacher–student distillation can be added later.
    """
    nn_model.train = True
    all_spikes = []

    for batch in tqdm(train_loader, desc="Training"):
        events, labels = batch
        # events shape: batch_size x N_events
        for ev, label in zip(events, labels):
            out_spikes = nn_model(ev)

            # TODO: define a loss
            # Right now, just record spikes
            all_spikes.append((out_spikes, label.item()))
    return all_spikes


def test_model(nn_model, test_loader):
    nn_model.eval = True
    correct = 0
    total = 0
    for batch in tqdm(test_loader, desc="Testing"):
        events, labels = batch
        for ev, label in zip(events, labels):
            out_spikes = nn_model(ev)

            # Example: classify by most active neuron in last layer
            if len(out_spikes) > 0:
                pred = np.bincount(out_spikes["n"]).argmax()
            else:
                pred = -1  # no spike = failure

            if pred == label.item():
                correct += 1
            total += 1
    return correct / total


if __name__ == "__main__":

    nn = model([
        # Assumed input layer 34x34
        Base_L(2*34*34),
        Neuron_L(4*34*34, mode="linear", layer_str="layer_1"),
        Neuron_L(2*34*34, mode="linear", layer_str="layer_2"),
        Neuron_L(10, mode="linear", layer_str="output_layer"),
    ])

    # Load dataset
    train_loader, test_loader = load_nmnist(batch_size=1)

    # Training loop (dummy for now)
    for epoch in range(1):
        spikes = train_one_epoch(nn, train_loader)
        acc = test_model(nn, test_loader)
        print(f"Epoch {epoch}: Test Accuracy = {acc:.4f}")
