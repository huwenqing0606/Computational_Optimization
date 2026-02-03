"""
CLI interfaces
"""

import click
from src.activations.plot_activations import plot_activations
from src.one_hidden_layer_nn.plot import plot_network_output, plot_network_loss


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


if __name__ == "__main__":
    cli()
