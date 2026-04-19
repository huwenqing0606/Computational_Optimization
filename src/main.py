"""
CLI interfaces
"""

import click
from src.activations.plot_activations import plot_activations
from src.one_hidden_layer_nn.plot import plot_network_output, plot_network_loss
from src.backpropagation.plotGD import execute_plot_gd_trajectory
from src.GD_AGD.plot import run_comparison
from src.SGD_variance_reduction.plot import plot_SGD_variance_reduction


@click.group()
def cli():
    pass


# activations
@cli.command(name="activations")
def activations():
    """
    activations
    """
    plot_activations()


# activations
@cli.command(name="one_hidden_layer_nn")
def one_hidden_layer_nn():
    """
    output and loss of one_hidden_layer_nn
    """
    plot_network_loss()
    plot_network_output()


# backpropagation
@cli.command(name="backpropagation")
def backpropagation():
    """
    GD trajectory via backpropagation gradient
    """
    execute_plot_gd_trajectory()


# GD-AGD
@cli.command(name="GD_AGD")
def GD_AGD():
    """
    GD vs AGD
    """
    run_comparison()


# GD-AGD
@cli.command(name="SGD_variance_reduction")
def SGD_variance_reduction():
    """
    SGD and variance reduction methods
    """
    plot_SGD_variance_reduction()


if __name__ == "__main__":
    cli()
