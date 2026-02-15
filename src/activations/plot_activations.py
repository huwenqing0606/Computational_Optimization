import os
import torch
import matplotlib.pyplot as plt

from src.activations.activations import Sigmoid, ReLU, Tanh, Exponential


ACTS = {
    "Sigmoid": Sigmoid,
    "ReLU": ReLU,
    "Tanh": Tanh,
    "Exponential": Exponential,
}
NAMELIST = ["Sigmoid", "ReLU", "Tanh", "Exponential"]

os.makedirs("output/activations", exist_ok=True)


def plot_activations():
    x = torch.linspace(-5, 5, 100, dtype=torch.float32)
    for name in NAMELIST:
        act = ACTS[name]()
        y = act.fn(x)
        dy = act.grad(x)
        if hasattr(act, "grad2"):
            d2y = act.grad2(x)
        else:
            d2y = torch.full_like(x, float("nan"))
        X = x.detach().cpu().numpy()
        Y = y.detach().cpu().numpy()
        Ygrad = dy.detach().cpu().numpy()
        Ygrad2 = d2y.detach().cpu().numpy()
        plt.plot(X, Y, label=r"$y$")
        plt.plot(X, Ygrad, label=r"$\frac{dy}{dx}$")
        plt.plot(X, Ygrad2, label=r"$\frac{d^2 y}{dx^2}$")
        plt.xlabel("x")
        plt.ylabel(f"y = {name}(x)")
        plt.legend()
        plt.savefig(f"output/activations/{name}.pdf", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    plot_activations()
