import numpy as np


class function_base:
    def __init__(self, name="base"):
        self.name = name

    def value(self, x_1, x_2):
        raise NotImplementedError

    def grad(self, x_1, x_2):
        raise NotImplementedError


"""
The quadratic function f and its gradients
f(x_1, x_2)=0.5 A x_1^2 + 0.5 B x_2^2
"""


class function_f(function_base):
    def __init__(self, axA, axB, name="f"):
        self.axA = axA
        self.axB = axB
        self.name = name

    def value(self, x_1, x_2):
        return 0.5 * self.axA * x_1 * x_1 + 0.5 * self.axB * x_2 * x_2

    def grad(self, x_1, x_2):
        return np.array([self.axA * x_1, self.axB * x_2])


"""
The perturbed quadratic function g and its gradients
g(x_1, x_2)=0.5 A x_1^2 + 0.5 B x_2^2+ epsilon(x_1^2+x_2^2)^{3/2}
"""


class function_g(function_base):
    def __init__(self, axA, axB, eps, name="g"):
        self.axA = axA
        self.axB = axB
        self.eps = eps
        self.name = name

    def value(self, x_1, x_2):
        return (
            0.5 * self.axA * x_1 * x_1
            + 0.5 * self.axB * x_2 * x_2
            + self.eps * ((np.sqrt(x_1 * x_1 + x_2 * x_2)) ** 3)
        )

    def grad(self, x_1, x_2):
        return np.array(
            [
                self.axA * x_1 + 3 * self.eps * x_1 * np.sqrt(x_1**2 + x_2**2),
                self.axB * x_2 + 3 * self.eps * x_2 * np.sqrt(x_1**2 + x_2**2),
            ]
        )


"""
The non-convex function h and its gradients
h(x_1, x_2)=0.5 A x_1^2 - 0.5 B x_2^2
"""


class function_h(function_base):
    def __init__(self, axA, axB, name="h"):
        self.axA = axA
        self.axB = axB
        self.name = name

    def value(self, x_1, x_2):
        return 0.5 * self.axA * x_1 * x_1 - 0.5 * self.axB * x_2 * x_2

    def grad(self, x_1, x_2):
        return np.array([self.axA * x_1, -self.axB * x_2])
