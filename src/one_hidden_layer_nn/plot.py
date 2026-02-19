import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.activations.activations import Sigmoid, ReLU, Tanh, Exponential
from src.one_hidden_layer_nn.network import OneHiddenLayerNetwork


ACTS = {"Sigmoid": Sigmoid, "ReLU": ReLU, "Tanh": Tanh, "Exponential": Exponential}
NAMELIST = ["Sigmoid", "ReLU", "Tanh", "Exponential"]

os.makedirs("output/one_hidden_layer_nn", exist_ok=True)


def plot_network_loss(
    layer_neuron_number=10000000, training_size=10, N=100, dtype=torch.float32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # fix weights
    weight_a_secondpart = torch.randn(
        layer_neuron_number - 2, device=device, dtype=dtype
    )
    weight_b = torch.randn(layer_neuron_number, 1, device=device, dtype=dtype)
    weight_c = torch.randn(1, layer_neuron_number, device=device, dtype=dtype)
    # set (x,y)
    X = torch.randn(1, training_size, device=device, dtype=dtype)
    Y = torch.randn(1, training_size, device=device, dtype=dtype)
    # a = (a_1, a_2, rest)
    a_1_tensor = torch.linspace(-10, 10, N, device=device, dtype=dtype)
    a_2_tensor = torch.linspace(-10, 10, N, device=device, dtype=dtype)
    # --- pre-compute fixed part ---
    # T = training_size
    # m = layer_neuron_number
    # X: (1,T)
    # weight_a_secondpart: (m-2,)
    # fixed_part: (m-2, T)
    fixed_part = weight_a_secondpart.view(-1, 1) @ X  # (m-2,1) @ (1,T) -> (m-2,T)
    # --- z buffer ---
    z = torch.empty(
        (layer_neuron_number, training_size), device=device, dtype=dtype
    )  # (m,T)
    # fix z[2:] = fixed_part - weight_b[2:]
    z[2:, :] = fixed_part - weight_b[2:, :]  # broadcast (m-2,1) -> (m-2,T)
    # extract b0 and b1
    b0 = weight_b[0:1, :]  # (1,1)
    b1 = weight_b[1:2, :]  # (1,1)
    # loop
    for name in NAMELIST:
        print(
            f"Calculating {name} landscape with"
            f" layer neuron number {layer_neuron_number}"
        )
        L_matrix = torch.empty((N, N), device=device, dtype=dtype)
        # instatiate activations
        act = ACTS[name]()
        # loop with a1 in outer loop and a2 inner loop
        with torch.no_grad():
            for i in range(N):
                a1 = a_1_tensor[i]
                # update row 0
                z[0:1, :] = a1 * X - b0
                for j in range(N):
                    a2 = a_2_tensor[j]
                    # update row 1
                    z[1:2, :] = a2 * X - b1
                    # calculate y_pred
                    a = act(z)
                    y_pred = weight_c @ a
                    loss = 0.5 * torch.mean((Y - y_pred) ** 2)
                    L_matrix[i, j] = loss
        # plot via cpu
        u = a_1_tensor.detach().cpu().numpy()
        v = a_2_tensor.detach().cpu().numpy()
        w = L_matrix.detach().cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        U, V = np.meshgrid(u, v, indexing="ij")
        ax.plot_surface(U, V, w, rstride=1, cstride=1, cmap="rainbow")
        ax.set_title(
            (
                f"{name} empirical loss landscape, "
                f"hidden layer size={layer_neuron_number}, "
                f"training size={training_size}"
            )
        )
        ax.set_zlabel("Empirical Loss")
        ax.set_xlabel("weight a_1")
        ax.set_ylabel("weight a_2")
        plt.savefig(
            (
                f"output/one_hidden_layer_nn/OneHiddenLayerNN-Loss_{name}"
                f"_layersize={layer_neuron_number}, "
                f"trainingsize={training_size}.jpg"
            ),
            bbox_inches="tight",
        )
        plt.close()


def plot_network_output(layer_neuron_number=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(-5, 5, 100, device=device)
    for name in NAMELIST:
        act = ACTS[name]()
        net = OneHiddenLayerNetwork(
            weight_a=np.random.randn(layer_neuron_number),
            weight_b=np.random.randn(layer_neuron_number),
            weight_c=np.random.randn(layer_neuron_number),
            activation=act,
        )
        y = net.output(x)
        X = x.detach().cpu().numpy()
        plt.plot(X, y, label=name)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(
            (
                f"One hidden layer neural network with {name} activation, "
                f"hidden layer size={layer_neuron_number}"
            )
        )
        plt.legend()
        plt.savefig(
            (
                f"output/one_hidden_layer_nn/OneHiddenLayerNN_{name}, "
                f"layersize={layer_neuron_number}.jpg"
            ),
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    plot_network_loss()
    plot_network_output()
