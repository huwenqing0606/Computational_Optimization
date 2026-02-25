from abc import ABC, abstractmethod
import torch
import numpy as np


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        pass

    @abstractmethod
    def grad(self, z):
        pass

    @abstractmethod
    def grad2(self, z):
        pass


class Sigmoid(ActivationBase):
    def fn(self, z):
        if isinstance(z, torch.Tensor):
            return torch.sigmoid(z)
        elif isinstance(z, (np.ndarray, np.generic)):
            return 1 / (1 + np.exp(-z))
        else:
            raise TypeError(f"Unsupported type {type(z)}")

    def grad(self, x):
        s = self.fn(x)
        return s * (1 - s)

    def grad2(self, x):
        s = self.fn(x)
        return s * (1 - s) * (1 - 2 * s)


class ReLU(ActivationBase):
    def __str__(self):
        return "ReLU"

    def fn(self, z):
        if isinstance(z, torch.Tensor):
            return torch.clamp(z, min=0)
        elif isinstance(z, (np.ndarray, np.generic)):
            return np.clip(z, 0, np.inf)
        else:
            raise TypeError(f"Unsupported type {type(z)}")

    def grad(self, x):
        if isinstance(x, torch.Tensor):
            return (x > 0).to(x.dtype)
        elif isinstance(x, (np.ndarray, np.generic)):
            return (x > 0).astype(x.dtype)
        else:
            raise TypeError(f"Unsupported type {type(x)}")

    def grad2(self, x):
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        elif isinstance(x, (np.ndarray, np.generic)):
            return np.zeros_like(x)
        else:
            raise TypeError(f"Unsupported type {type(x)}")


class Tanh(ActivationBase):
    def __str__(self):
        return "Tanh"

    def fn(self, z):
        if isinstance(z, torch.Tensor):
            return torch.tanh(z)
        elif isinstance(z, (np.ndarray, np.generic)):
            return np.tanh(z)
        else:
            raise TypeError(f"Unsupported type {type(z)}")

    def grad(self, x):
        s = self.fn(x)
        return 1 - s**2

    def grad2(self, x):
        s = self.fn(x)
        return -2 * s * (1 - s**2)


class Exponential(ActivationBase):
    def __str__(self):
        return "Exponential"

    def fn(self, z):
        if isinstance(z, torch.Tensor):
            return torch.exp(z)
        elif isinstance(z, (np.ndarray, np.generic)):
            return np.exp(z)
        else:
            raise TypeError(f"Unsupported type {type(z)}")

    def grad(self, x):
        return self.fn(x)

    def grad2(self, x):
        return self.fn(x)
