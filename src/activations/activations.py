from abc import ABC, abstractmethod

import numpy as np


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        raise NotImplementedError


class Sigmoid(object):
    def __init__(self):
        super().__init__()

    def fn(self, z):
        return 1 / (1 + np.exp(-z))

    def grad(self, x):
        fn_x = self.fn(self, x)
        return fn_x * (1 - fn_x)

    def grad2(self, x):
        fn_x = self.fn(self, x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class ReLU(object):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    def fn(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, x):
        return (x > 0).astype(int)

    def grad2(self, x):
        return np.zeros_like(x)


class Tanh(object):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Tanh"

    def fn(self, z):
        return np.tanh(z)

    def grad(self, x):
        return 1 - np.tanh(x) ** 2

    def grad2(self, x):
        tanh_x = np.tanh(x)
        return -2 * tanh_x * (1 - tanh_x**2)


class Exponential(object):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Exponential"

    def fn(self, z):
        return np.exp(z)

    def grad(self, x):
        return np.exp(x)

    def grad2(self, x):
        return np.exp(x)
