from src.SGD_variance_reduction.size import size
import numpy as np


class LossFunction:
    """
    Loss Function L(w_1, w_2; (x_1, x_2, y)) = 0.5(Aw_1x_1+Bw_2x_2-y)^2 for A, B>0
    and its gradients with respect to the weight parameters w_1 and w_2
    the gradients are calculated via the tf.GradientTape() mode
    """

    def __init__(self, axA, axB):
        self.axA = axA
        self.axB = axB

    # value of the loss function
    def value(self, w, x, y):
        return 0.5 * (self.axA * w[0] * x[0] + self.axB * w[1] * x[1] - y) ** 2

    # gradient of the loss function with respect to the weights (w_1, w_2)
    def grad(self, w, x, y):
        residual = self.axA * w[0] * x[0] + self.axB * w[1] * x[1] - y
        return np.array(
            [residual * self.axA * x[0], residual * self.axB * x[1]], dtype=float
        )

    # average of a sequence of function=loss functions/loss gradients for
    #   a given list of samples (x_i, y_i)
    def average(self, w, sample_x, sample_y, function):
        sample_size = size(sample_x, sample_y)
        total = 0.0
        for i in range(sample_size):
            total += function(w, sample_x[i], sample_y[i])
        return total / sample_size
