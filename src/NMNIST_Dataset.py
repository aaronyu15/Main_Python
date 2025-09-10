from torch.utils.data import Dataset
import random
import tonic
import tonic.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class NMNIST_Dataset(Dataset):
    def __init__(self, base_dataset, allowed_classes, max_samples=None):
        self.base_dataset = base_dataset
        self.indices = [i for i, (_, y) in enumerate(base_dataset) if y in allowed_classes]
        if max_samples is not None:
            self.indices = random.sample(self.indices, max_samples)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]



def play_frames(frames, frame_transform=False):
    fig, ax = plt.subplots()
    if not frame_transform:
        im = ax.imshow(frames[-1][1] - frames[0][0])
    else:
        im = ax.imshow(frames[-1])
    ax.axis("off")

    def update(frame_data):
        if not frame_transform:
            im.set_data(frame_data[0] - frame_data[0])
        else:
            im.set_data(frame_data)
        return [im]

   # Create the animation
    ani = FuncAnimation(fig, update, frames=frames, interval=9)

    plt.tight_layout()
    plt.show()

def show_frame(frames, index):
   plt.imshow(frames[index])
   plt.axis("off")
   plt.show()

if __name__ == "__main__":

    dataset = tonic.datasets.NMNIST(save_to="./dataset/nmnist", train=False)

    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, time_window=3000)
    
    events, target = dataset[3000]

    frames = frame_transform(events)
    print(frames.shape)

    # Play all frames as an animation
    # Note: for frame_transform, set to True if using ToFrame transform
    frame_transform = True
    if frame_transform:
        frames = abs(frames[:, -1, :, :]) + abs(frames[:, 1, :, :])
        frames = frames.clip(-1, 1)
    play_frames(frames, frame_transform=frame_transform)
