# Nesterov for qudratic functions and perturbed quadratic functions

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from src.GD_AGD.functions import function_f
from src.GD_AGD.optimizers import Optimizers
import os


path = "output/GD_AGD"
os.makedirs(path, exist_ok=True)


def make_tag(optname, A, B, alpha, beta, epsilon):
    return f"{optname}_A={A}_B={B}_alpha={alpha}_beta={beta}_eps={epsilon}"


def plot_landscape(
    function,
    optname,
    A,
    B,
    alpha,
    beta,
    epsilon,
    xlim=(-10, 10),
    ylim=(-10, 10),
    grid_n=100,
    save=True,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(xlim[0], xlim[1], grid_n)
    v = np.linspace(ylim[0], ylim[1], grid_n)
    u, v = np.meshgrid(u, v)
    w = function.value(u, v)
    ax.plot_surface(u, v, w, rstride=1, cstride=1, cmap="summer")
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_zlabel(function.name + "(x_1, x_2)")
    if save:
        print(f"saving Landscape_{make_tag(optname, A, B, alpha, beta, epsilon)}.jpg")
        plt.savefig(
            path + f"/Landscape_{make_tag(optname, A, B, alpha, beta, epsilon)}.jpg"
        )
    plt.close()


def plot_trajectory_3d(trajectory_x_1, trajectory_x_2, loss, function, optname):
    fig = plt.figure()
    mpl.rcParams["legend.fontsize"] = 10
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        trajectory_x_1, trajectory_x_2, loss, label=optname + " trajectory", color="r"
    )
    ax.legend()
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_zlabel(function.name + "(x_1, x_2)")
    plt.close()


def plot_trajectory_2d(trajectory_x_1, trajectory_x_2, optname):
    plt.figure()
    plt.plot(trajectory_x_1, trajectory_x_2)
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.title(optname)
    plt.close()


def plot_loss(loss, optname, A, B, alpha, beta, epsilon, save=True):
    plt.figure()
    plt.plot(loss)
    plt.xlabel("iteration")
    plt.ylabel("function error to minimum")
    plt.title(optname)
    if save:
        print(f"saving Loss_{make_tag(optname, A, B, alpha, beta, epsilon)}.jpg")
        plt.savefig(path + f"/Loss_{make_tag(optname, A, B, alpha, beta, epsilon)}.jpg")
    plt.close()


def plot_distance(distance, optname, A, B, alpha, beta, epsilon, save=True):
    plt.figure()
    plt.plot(distance)
    plt.xlabel("iteration")
    plt.ylabel("distance to minimizer")
    plt.title(optname)
    if save:
        print(
            (
                f"saving "
                f"Distance_To_Zero_{make_tag(optname, A, B, alpha, beta, epsilon)}.jpg"
            )
        )
        plt.savefig(
            path
            + f"/Distance_To_Zero_{make_tag(optname, A, B, alpha, beta, epsilon)}.jpg"
        )
    plt.close()


def animate_trajectory_3d(
    trajectory_x_1,
    trajectory_x_2,
    loss,
    optname,
    A,
    B,
    alpha,
    beta,
    epsilon,
    interval=10,
    save=True,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def init():
        ax.clear()
        return []

    def anmi(i):
        ax.clear()
        ax.plot(trajectory_x_1[:i], trajectory_x_2[:i], loss[:i], "b:")
        ax.plot(
            trajectory_x_1[max(i - 1, 0) : i],
            trajectory_x_2[max(i - 1, 0) : i],
            loss[max(i - 1, 0) : i],
            "bo",
            markersize=10,
        )
        ax.set_xlabel("x_1")
        ax.set_ylabel("x_2")
        ax.set_zlabel("loss")
        return []

    anim = animation.FuncAnimation(
        fig,
        anmi,
        init_func=init,
        frames=len(loss),
        interval=interval,
        blit=False,
        repeat=False,
    )
    if save:
        print(f"saving Dynamics_{make_tag(optname, A, B, alpha, beta, epsilon)}.gif")
        anim.save(
            path + f"/Dynamics_{make_tag(optname, A, B, alpha, beta, epsilon)}.gif",
            writer="pillow",
        )


def plot_all_results(
    trajectory_x_1,
    trajectory_x_2,
    loss,
    distance,
    function,
    optname,
    A,
    B,
    alpha,
    beta,
    epsilon,
    save=True,
    make_gif=True,
):
    plot_landscape(function, optname, A, B, alpha, beta, epsilon, save=save)
    plot_trajectory_3d(trajectory_x_1, trajectory_x_2, loss, function, optname)
    plot_trajectory_2d(trajectory_x_1, trajectory_x_2, optname)
    plot_loss(loss, optname, A, B, alpha, beta, epsilon, save=save)
    plot_distance(distance, optname, A, B, alpha, beta, epsilon, save=save)
    if make_gif:
        animate_trajectory_3d(
            trajectory_x_1,
            trajectory_x_2,
            loss,
            optname,
            A,
            B,
            alpha,
            beta,
            epsilon,
            save=save,
        )


def run_optimizer(
    function,
    optimization,
    optname,
    x_init=None,
    steps=1000,
    lr=0.01,
    alpha=0.01,
    beta=0.0,
):
    if x_init is None:
        x_current = np.random.uniform(-1, 1, size=2)
    else:
        x_current = np.array(x_init, dtype=float)
    x_current_minus1 = x_current.copy()
    trajectory_x_1 = []
    trajectory_x_2 = []
    loss = []
    distance = []
    for _ in range(steps):
        trajectory_x_1.append(x_current[0])
        trajectory_x_2.append(x_current[1])
        loss.append(function.value(x_current[0], x_current[1]))
        distance.append(
            np.sqrt(x_current[0] * x_current[0] + x_current[1] * x_current[1])
        )
        x = x_current + optimization.update(
            x_current[0],
            x_current[1],
            x_current_minus1[0],
            x_current_minus1[1],
            lr,
            alpha,
            beta,
            optname,
        )
        x_current_minus1 = x_current.copy()
        x_current = x
    return trajectory_x_1, trajectory_x_2, loss, distance


def plot_optimizer(A, B, epsilon, function, optname):
    m = min(A, B)
    L = max(A, B)
    kappa = L / m
    if optname == "Nesterov":
        alpha = 1 / L
        beta = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    elif optname == "HeavyBall":
        alpha = 4 / (np.sqrt(L) + np.sqrt(m)) ** 2
        beta = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    else:
        raise ValueError
    optimization = Optimizers(function=function)
    trajectory_x_1, trajectory_x_2, loss, distance = run_optimizer(
        function=function,
        optimization=optimization,
        optname=optname,
        steps=1000,
        lr=1 / L,
        alpha=alpha,
        beta=beta,
    )
    plot_all_results(
        trajectory_x_1,
        trajectory_x_2,
        loss,
        distance,
        function,
        optname,
        A,
        B,
        alpha,
        beta,
        epsilon,
    )


def run_comparison(
    A=1, B=100, epsilon=0.1, lr=0.01, alpha=0.01, beta=1, function=function_f(1, 100)
):
    for optname in ["GD", "HeavyBall", "Nesterov"]:
        optimization = Optimizers(function=function)
        trajectory_x_1, trajectory_x_2, loss, distance = run_optimizer(
            function=function,
            optimization=optimization,
            optname=optname,
            steps=1000,
            lr=lr,
            alpha=alpha,
            beta=beta,
        )
        plot_all_results(
            trajectory_x_1,
            trajectory_x_2,
            loss,
            distance,
            function,
            optname,
            A,
            B,
            alpha,
            beta,
            epsilon,
        )
