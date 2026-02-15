from abc import ABC, abstractmethod
import torch


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        raise NotImplementedError

    @abstractmethod
    def grad(self, z):
        raise NotImplementedError


class Sigmoid(ActivationBase):
    def fn(self, z):
        return torch.sigmoid(z)

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
        return torch.clamp(z, min=0)

    def grad(self, x):
        return (x > 0).to(x.dtype)

    def grad2(self, x):
        return torch.zeros_like(x)


class Tanh(ActivationBase):
    def __str__(self):
        return "Tanh"

    def fn(self, z):
        return torch.tanh(z)

    def grad(self, x):
        return 1 - torch.tanh(x) ** 2

    def grad2(self, x):
        t = torch.tanh(x)
        return -2 * t * (1 - t**2)


class Exponential(ActivationBase):
    def __str__(self):
        return "Exponential"

    def fn(self, z):
        return torch.exp(z)

    def grad(self, x):
        return torch.exp(x)

    def grad2(self, x):
        return torch.exp(x)
