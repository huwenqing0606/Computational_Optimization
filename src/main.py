"""
CLI interfaces
"""

import click
from src.activations.plot_activations import plot_activations


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


if __name__ == "__main__":
    cli()
