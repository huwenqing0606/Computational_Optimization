"""
one_hidden_layer_nn.network
"""

import torch


class OneHiddenLayerNetwork:
    def __init__(self, weight_a, weight_b, weight_c, activation):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_a = torch.tensor(
            weight_a, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        self.weight_b = torch.tensor(
            weight_b, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        self.weight_c = torch.tensor(
            weight_c, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        self.activation = activation

    def output(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device).reshape(1, -1)
        with torch.no_grad():  # inference mode
            z = torch.matmul(self.weight_a, x) - self.weight_b
            a = self.activation(z)
            y = torch.matmul(self.weight_c, a)
        return y.cpu().numpy().flatten()
