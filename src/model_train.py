# model_train.py
import numpy as np
import tonic
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging

from SNNModel import model # import your model definition
from SNNLogger import SNNLogger  # for logging training progress
from NMNIST_Dataset import NMNIST_Dataset  # custom dataset wrapper
from train_utils import *
import utils


def collate_frames(batch):
    frames, labels = zip(*batch)
    max_T = max(f.shape[0] for f in frames)
    P, X, Y = frames[0].shape[1:]
    batch_frames = np.zeros((len(frames), max_T, P, X, Y), dtype=frames[0].dtype)
    for i, f in enumerate(frames):
        batch_frames[i, :f.shape[0]] = f  # pad with zeros
    return batch_frames, np.array(labels)

def collate_events(batch):
    # batch is list of (events, label)
    events = [b[0] for b in batch]  # keep as list of numpy arrays
    labels = [b[1] for b in batch]
    return events, np.array(labels)

def load_nmnist(batch_size=1, samples=100, classes={0,1,2}):
    """
    Load NMNIST dataset with tonic.
    Converts events into numpy arrays with fields x, y, t, p.
    """

    train_dataset = tonic.datasets.NMNIST(save_to="../dataset/nmnist", train=True , transform=None)
    test_dataset  = tonic.datasets.NMNIST(save_to="../dataset/nmnist", train=False, transform=None)

    # Filter dataset to only include specified classes and limit samples
    train_dataset = NMNIST_Dataset(train_dataset, allowed_classes=classes, samples_per_class=samples, use_frames=True, frame_time_window=3000, frame_clip=True)
    test_dataset  = NMNIST_Dataset(test_dataset , allowed_classes=classes, samples_per_class=samples, use_frames=True, frame_time_window=3000, frame_clip=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True , collate_fn=collate_frames)
    test_loader  = DataLoader(test_dataset , batch_size=batch_size, shuffle=False, collate_fn=collate_frames)

    return train_loader, test_loader


def train_one_epoch(nn, train_loader, num_classes=5, lr=0.01, epoch_num=0, debug=False, logger=None):
    total_loss, correct, total = 0, 0, 0
    num_layers = len(nn.layers["membrane_potential_map"])

    for b, (frames_batch, labels) in enumerate(tqdm(train_loader, desc="Training")):
        batch_size = len(labels)

        # Accumulate grads for this batch
        grads = init_grads(nn)

        # ---- Forward pass (all samples at once) ----
        out_spikes_batch, activations = nn(frames_batch, return_intermediates=True)  

        # out_spikes_batch is a list of length B
        spike_counts_batch = np.zeros((batch_size, num_classes), dtype=np.int64)
        for i, out_spikes in enumerate(out_spikes_batch):
            spike_counts_batch[i] = np.bincount(out_spikes['n'], minlength=num_classes)


        # Loss
        losses, error_grad_out_batch = batch_loss_fn(spike_counts_batch, labels)
        total_loss += np.sum(losses)

        # Prediction
        preds = np.argmax(spike_counts_batch, axis=1)
        correct += np.sum(preds == labels)
        total += batch_size 

        # ---- Backpropagation ----
        next_error_batch = error_grad_out_batch

        for l in reversed(range(num_layers)):  # loop over all layers backwards
            # presynaptic activations: activations[l-1]
            if l == 0:
                batch_activations = frames_batch.reshape(batch_size, -1)  # flatten raw input
            else:
                batch_activations = activations[l-1]

            grads = compute_layer_grads_batch(grads, nn, l, next_error_batch, batch_activations)

            if l > 0:
                next_error_batch = backprop_error_batch(nn.layers, l, next_error_batch)

        if debug:
            logger.info(f"Epoch [{epoch_num}] Batch [{b}]: Loss={total_loss:.4f}, Train Acc={correct / total:.4f}")
        

        update_layer_params(grads, nn, batch_size=batch_size)


    return total_loss / total, correct / total






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

def get_args():
    parser = argparse.ArgumentParser(description="Train XOR-based SNN on N-MNIST")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--run_train"  , action="store_true", default=False, help="Whether to run testing")
    parser.add_argument("--run_test"   , action="store_true", default=False, help="Whether to run testing")
    parser.add_argument("--run_sample" , action="store_true", default=False, help="Whether to run testing")

    # Dataset args
    parser.add_argument("--classes", type=int, nargs="+", default=[0,1,2,3,4], help="Which digit classes to use")
    parser.add_argument("--samples", type=int, default=100, help="Number of per-class samples to use from dataset")

    # Training args
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")

    # Model args
    parser.add_argument("--save-path", type=str, default="../checkpoints/model.npz", help="Where to save trained model")

    parser.add_argument("--log-file", type=str, default="training.log", help="Where to save trained model")
    parser.add_argument("--debug"   , action="store_true", default=False, help="Whether to run testing")
    return parser.parse_args()

if __name__ == "__main__":
    """
    python model_train.py --run_train 
    
    """

    args = get_args()

    np.random.seed(args.seed)

    hidden_sizes = (4*34*34, 2*34*34)  # example hidden layer sizes

    nn = model(
        x=34, 
        y=34, 
        p=2, 
        hidden_sizes=hidden_sizes, 
        out_size=10,
    )

    logger = SNNLogger(log_level=logging.DEBUG, log_file=args.log_file)
    logger.record_args(args)

    if args.debug:
        logger.setup_monitoring(nn, choice="select", num_per_layer=1)
        nn.set_logger(logger)


    if args.run_train:
        logger.info("Starting training...")
        
        # Load dataset
        train_loader, test_loader = load_nmnist(batch_size=args.batch_size, samples=args.samples, classes=set(args.classes))

        # Training loop
        for epoch in range(args.epochs):
            avg_loss, train_acc = train_one_epoch(nn, train_loader, num_classes=len(args.classes), lr=args.lr, epoch_num=epoch, debug=args.debug, logger=logger)
            test_acc = test_model(nn, test_loader)

            logger.info(f"Epoch [{epoch}]: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc{test_acc:.4f}")

        # Save model parameters
        model.save(path=args.save_path)
        print(f"Model saved to {args.save_path}")

    elif args.run_test:
        logger.info("Starting testing...")

        # Load dataset
        _, test_loader = load_nmnist(batch_size=args.batch_size, samples=args.samples, classes=set(args.classes))

        # Load model parameters
        nn.load(path=args.save_path)

        # Test
        test_acc = test_model(nn, test_loader)
        print(f"Test Accuracy: {test_acc:.4f}")

    elif args.run_sample:
        logger.info("Running a sample...")
        # Load model parameters
        _, test_loader = load_nmnist(batch_size=1, samples=1, classes=set(args.classes))

        nn.load(path=args.save_path)

        # Sample inference
        sample = next(iter(test_loader))
        sample_events, label =  sample # get events of first sample

        logger.info(f"Sample true label: {label}")
        logger.info(f"Sample number of timesteps: {sample_events.shape[1]}")
        logger.info(f"Sample shape: {sample_events.shape}")

        out_spikes = nn(sample_events, debug=args.debug)

        logger.save_activity()
        logger.plot_membrane_traces("Layer0")

        print("Output spikes:", out_spikes[0])
        utils.plot_raster(out_spikes[0], title=f"Output Spikes for Label {label}")

        
