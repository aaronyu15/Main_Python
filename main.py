import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dataset = tonic.datasets.NMNIST(save_to="./dataset/nmnist", train=False)

sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, time_window=3000)


events, target = dataset[3000]

print(target)
print(len(events['t']))
print(max(events['x']))
print(max(events['y']))

frames = frame_transform(events)

print(frames.shape)

def play_frames(frames, frame_transform=False):
    fig, ax = plt.subplots()
    if not frame_transform:
        im = ax.imshow(frames[0][1] - frames[0][0])
    else:
        im = ax.imshow(frames[0])
    ax.axis("off")

    def update(frame_data):
        if not frame_transform:
            im.set_data(frame_data[1] - frame_data[0])
        else:
            im.set_data(frame_data)
        return [im]

   # Create the animation
    ani = FuncAnimation(fig, update, frames=frames, interval=10)

    plt.tight_layout()
    plt.show()

def show_frame(frames, index):
   plt.imshow(frames[index])
   plt.axis("off")
   plt.show()

if __name__ == "__main__":
    # Show a single frame
    #show_frame(frames, 10)

    # Play all frames as an animation
    # Note: for frame_transform, set to True if using ToFrame transform
    frame_transform = True
    if frame_transform:
        frames = abs(frames[:, 0, :, :]) + abs(frames[:, 1, :, :])
        frames = frames.clip(0, 1)
    play_frames(frames, frame_transform=frame_transform)
