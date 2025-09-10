
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from random import random
import numpy as np
import math
import matplotlib.pyplot as plt
from model import model, Base_L, Neuron_L
from utils import plot_raster

if __name__ == "__main__":

    dataset = tonic.datasets.NMNIST(save_to="./dataset/nmnist", train=False)
    # Events have x, y, t, p ordering
    # Max x is 33, max y is 33
    # t goes up to around 300000
    events, target = dataset[0]


    print(f"Target: {target}")
    nn = model([
        # Assumed input layer 34x34
        Base_L(2*34*34),
        Neuron_L(4*34*34, mode="linear", layer_str="layer_1"),
        Neuron_L(2*34*34, mode="linear", layer_str="layer_4"),
        Neuron_L(10, mode="linear", layer_str="output_layer"),
    ])
        
    out_spikes = nn(events)
    print(f"Output spikes: {out_spikes}")
    print(f"Output spikes shape: {out_spikes.shape}")

    plot_raster(out_spikes, title="Output Layer Spikes", marker_size=2)

    exit()

    epochs = 200
    alpha = 0.01

    num_input = 5

    n = neuron(alpha=alpha, mode="linear")


    target_y = [-22.0, 12.2, 2.3]

    py_list = []

    print(f"Target: {target_y}")

    for i in range(epochs):
        idx = np.random.randint(0, 3)
        target = target_y[idx]
        input_x = np.zeros(3)
        input_x[idx] = 1

        predict_y = n(input_x)  # Assuming input is
        n.backward(target_y=target, predict_y=predict_y, idx=idx)

        py_list.append(predict_y)
        print(f"Updated prediction: {predict_y}")

    fig = plt.figure()
    x_vals = range(epochs)

    in_x = [1, 1, 0]
    predict_y = n(in_x)
    plt.scatter(x_vals[-1]+1, predict_y, label='Pred')

    in_x = [1, 0, 1]
    predict_y = n(in_x)
    plt.scatter(x_vals[-1]+2, predict_y, label='Pred')
    
    in_x = [0, 1, 1]
    predict_y = n(in_x)
    plt.scatter(x_vals[-1]+3, predict_y, label='Pred')

    in_x = [1, 1, 1]
    predict_y = n(in_x)
    plt.scatter(x_vals[-1]+4, predict_y, label='Pred')


    plt.scatter(x_vals, np.full_like(x_vals, target_y[0]), label='True')
    plt.scatter(x_vals, np.full_like(x_vals, target_y[1]), label='True')
    plt.scatter(x_vals, np.full_like(x_vals, target_y[2]), label='True')

    plt.scatter(x_vals, py_list, label='Pred')

    plt.title('Model Prediction vs Target')
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.legend()
    plt.grid(True)
    plt.show()
