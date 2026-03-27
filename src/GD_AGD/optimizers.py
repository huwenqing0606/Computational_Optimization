import numpy as np
from src.GD_AGD.functions import function_base

"""
The optimizer update for: GD, Heavy-Ball, Nesterov
"""


class Optimizers:
    def __init__(self, function: function_base):
        self.function = function

    def GD(self, x_1, x_2, lr):
        grad = self.function.grad(x_1, x_2)
        update = -lr * grad
        return update

    def HeavyBall(self, x_1, x_2, x_1_old, x_2_old, alpha, beta):
        grad = self.function.grad(x_1, x_2)
        momentum = np.array([x_1 - x_1_old, x_2 - x_2_old])
        return -alpha * grad + beta * momentum

    def Nesterov(self, x_1, x_2, x_1_old, x_2_old, alpha, beta):
        grad = self.function.grad(
            x_1 + beta * (x_1 - x_1_old), x_2 + beta * (x_2 - x_2_old)
        )
        momentum = np.array([x_1 - x_1_old, x_2 - x_2_old])
        return -alpha * grad + beta * momentum

    def update(self, x_1, x_2, x_1_old, x_2_old, lr, alpha, beta, method):
        update = np.array([0, 0])
        if method == "GD":
            update = self.GD(x_1, x_2, lr)
        elif method == "HeavyBall":
            update = self.HeavyBall(x_1, x_2, x_1_old, x_2_old, alpha, beta)
        elif method == "Nesterov":
            update = self.Nesterov(x_1, x_2, x_1_old, x_2_old, alpha, beta)
        else:
            update = np.array([0, 0])
        return update
