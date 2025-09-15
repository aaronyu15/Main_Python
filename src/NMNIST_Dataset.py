from torch.utils.data import Dataset
import random
import tonic
import tonic.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class NMNIST_Dataset(Dataset):
    def __init__(self, base_dataset, allowed_classes, samples_per_class=None, use_frames=False, frame_time_window=3000, frame_clip=True):
        self.base_dataset = base_dataset
        self.allowed_classes = allowed_classes
        self.max_samples = samples_per_class
        self.use_frames = use_frames
        self.frame_clip = frame_clip


        class_to_indices = {c: [] for c in allowed_classes}
        for i, (_, label) in enumerate(base_dataset):
            if label in allowed_classes:
                class_to_indices[label].append(i)
                
        self.indices = []
        if samples_per_class is not None:
            for c in allowed_classes:
                available = class_to_indices[c]
                n_pick = min(samples_per_class, len(available))
                self.indices.extend(random.sample(available, n_pick))
        else:
            for c in allowed_classes:
                self.indices.extend(class_to_indices[c])

        if use_frames:
            sensor_size = tonic.datasets.NMNIST.sensor_size
            self.frame_transform = transforms.ToFrame(sensor_size=sensor_size, time_window=frame_time_window)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        events, label = self.base_dataset[self.indices[idx]]
        if self.use_frames:
            frames = self.frame_transform(events).astype(np.uint8)
            if self.frame_clip:
                frames = np.clip(frames, 0, 1)  # clip extreme values for stability
            return frames, label

        else:
            return events, label
    
    def get_largest_time(self):
        max_t = 0
        for i in range(len(self.base_dataset)):
            events, _ = self.base_dataset[i]
            max_t = max(max_t, max(events['t']))
        return max_t



def play_frames(frames, single_frame=False, frame_idx=0, clip_frames=False):
    fig, ax = plt.subplots()
    
    # Reduce to 2D by summing over polarity, only for visualization
    if frames.ndim == 4:  # (T, P, X, Y)
        frames2d = frames.sum(axis=1)  # sum over P
    elif frames.ndim == 3:  # (P, X, Y)
        frames2d = frames.sum(axis=0)  # sum over P
    else:
        raise ValueError(f"Unexpected frame shape {frames.shape}")
    

    if clip_frames:
        frames2d = frames2d.clip(-1, 1)
    ax.axis("off")

    im = ax.imshow(frames2d[frame_idx if single_frame else 0], cmap='gray')

    def update(i):
        frame_data = frames2d[i]
        if clip_frames:
            frame_data = frame_data.clip(-1, 1)
        im.set_data(frame_data)
        return [im]

    # Create the animation
    if not single_frame:
        ani = FuncAnimation(fig, update, frames=len(frames2d), interval=50)

    plt.tight_layout()
    plt.show()


def plot_frame_histogram(frames, bins=50, title="Frame Value Histogram"):
    """
    Plot histogram of all values in a frame tensor (T, P, X, Y).
    
    Args:
        frames: np.ndarray of shape (T, P, X, Y)
        bins: number of histogram bins
        title: plot title
    """

    plt.figure(figsize=(7,5))

    values = frames.flatten()
    counts, edges, bars = plt.hist(values, bins=bins, color="steelblue", edgecolor="black")
    for rect, count in zip(bars, counts):
        if count > 0:
            plt.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                     f"{int(count)}", ha="center", va="bottom", fontsize=12)

    plt.hist(values, bins=bins, color="steelblue", edgecolor="black")
    plt.yscale("log")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    NMNIST = tonic.datasets.NMNIST(save_to="../dataset/nmnist", train=False)

    dataset = NMNIST_Dataset(NMNIST, allowed_classes=[0,1,2,3,4], samples_per_class=10, use_frames=True, frame_time_window=3000, frame_clip=True)
    
    frames, target = dataset[0]

    print(f"Shape of frames (T, P, X, Y): {frames.shape}")
    print(f"Largest time in dataset: {dataset.get_largest_time()}")

    play_frames(frames, single_frame=True, frame_idx=1, clip_frames=False)
    plot_frame_histogram(frames, bins=50, title="Frame Value Histogram")
