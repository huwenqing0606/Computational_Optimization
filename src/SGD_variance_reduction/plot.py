import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
import numpy as np
from src.SGD_variance_reduction.loss_function import LossFunction
from src.SGD_variance_reduction.stochastic_optimizers import stochastic_optimizer
import os


os.makedirs("output/SGD_variance_reduction", exist_ok=True)
path = "output/SGD_variance_reduction/"

# set the parameters A, B for the loss function
A = 1
B = 1

# set the training sample size and the batchsize
training_sample_size = 10
batchsize = 1

# for SGD, set the number of iteration steps
num_steps = 1000

# for SVRG, set the number of epochs and epochlength (m)
num_epochs = 100
epochlength = 10

# set the learning rate
lr = 0.01


def make_tag(optname: str) -> str:
    if optname == "SGD":
        return (
            f"{optname}_A={A}_B={B}_trainingsize={training_sample_size}"
            f"_batchsize={batchsize}_learningrate={lr}_steps={num_steps}"
        )
    elif optname in ["SVRG", "SARAH"]:
        return (
            f"{optname}_A={A}_B={B}_trainingsize={training_sample_size}"
            f"_learningrate={lr}_epochs={num_epochs}_epochlength={epochlength}"
        )
    else:
        raise ValueError(f"Unknown optname: {optname}")


"""
running the code, plot
(1) the trajectory animation;
(2) the evolution of the training error;
(3) the evolution of test error.
"""


def plot_an_opt(optname: str):
    # generate the training samples (x_i, y_i)
    training_sample_x = np.random.normal(0, 1, size=(training_sample_size, 2))
    training_sample_y = np.random.normal(0, 1, size=training_sample_size)

    # initialize the initial weights
    w_init = np.array([1.0, 1.0])

    # pick one test sample
    test_sample_x = np.random.normal(0, 1, size=2)
    test_sample_y = np.random.normal(0, 1)

    # set the loss function and the stochastic optimizer
    function = LossFunction(axA=A, axB=B)
    optimizer = stochastic_optimizer(
        function=function,
        training_sample_x=training_sample_x,
        training_sample_y=training_sample_y,
        test_sample_x=test_sample_x,
        test_sample_y=test_sample_y,
    )

    starttime = datetime.datetime.now()
    print(f"=== starting {optname} iteation and record cpu time ===")
    # optimize
    if optname == "SGD":
        trajectory_w, loss_list, test_error_list = optimizer.SGD_optimizer(
            w_init, num_steps, lr, batchsize
        )
    elif optname == "SVRG":
        trajectory_w, loss_list, test_error_list = optimizer.SVRG_optimizer(
            w_init, num_epochs, epochlength, lr
        )
    elif optname == "SARAH":
        trajectory_w, loss_list, test_error_list = optimizer.SARAH_optimizer(
            w_init, num_epochs, epochlength, lr
        )
    else:
        raise ValueError(f"optname = {optname} not SGD, SVRG or SARAH")

    endtime = datetime.datetime.now()
    print(f"=== finished {optname} iteation and report running time ===")
    print("running time:", (endtime - starttime).seconds, "seconds.")

    file_path = os.path.join(path, "records_" + make_tag(optname) + ".txt")
    with open(file_path, "w") as f:
        print("running time (seconds):", (endtime - starttime).seconds, file=f)
        print("weight trajectory =", trajectory_w, file=f)
        print("loss =", loss_list, file=f)
        print("test error =", test_error_list, file=f)

    # convert trajectory to arrays for plotting
    trajectory_w = np.array(trajectory_w, dtype=float)
    trajectory_w_1 = trajectory_w[:, 0]
    trajectory_w_2 = trajectory_w[:, 1]
    loss_arr = np.array(loss_list, dtype=float)

    # -----------------------------
    # 3D animation of the trajectory
    # -----------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def init():
        ax.clear()
        ax.set_xlabel("w_1")
        ax.set_ylabel("w_2")
        ax.set_zlabel("training loss")
        ax.set_title(optname)
        return []

    def anmi(i):
        ax.clear()
        ax.plot(
            trajectory_w_1[: i + 1], trajectory_w_2[: i + 1], loss_arr[: i + 1], "b:"
        )
        ax.plot(
            [trajectory_w_1[i]], [trajectory_w_2[i]], [loss_arr[i]], "bo", markersize=8
        )
        ax.set_xlabel("w_1")
        ax.set_ylabel("w_2")
        ax.set_zlabel("training loss")
        ax.set_title(optname)
        return []

    anim = animation.FuncAnimation(
        fig,
        anmi,
        init_func=init,
        frames=len(trajectory_w),
        interval=50,
        blit=False,
        repeat=False,
    )

    anim.save(os.path.join(path, make_tag(optname) + ".gif"), writer="pillow")

    # -----------------------------
    # plot training loss
    # -----------------------------
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title(optname)
    plt.savefig(os.path.join(path, "Loss_" + make_tag(optname) + ".jpg"))

    # -----------------------------
    # plot test error
    # -----------------------------
    plt.figure()
    plt.plot(test_error_list)
    plt.xlabel("iteration")
    plt.ylabel("test error")
    plt.title(optname)
    plt.savefig(os.path.join(path, "TestError_" + make_tag(optname) + ".jpg"))


"""
plot all and compare
"""


def plot_SGD_variance_reduction():
    plot_an_opt("SGD")
    plot_an_opt("SVRG")
    plot_an_opt("SARAH")
