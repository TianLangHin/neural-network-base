from types import AllParams, BackwardErrorCache, Layer, ParamSet
from typing import List
import numpy as np

class NeuralNetwork:

    @staticmethod
    def xavier_normal(
            rng: np.random.Generator,
            features_in: int,
            features_out: int,
            *,
            size: List[int]) -> np.ndarray:
        stdev = np.sqrt(2 / (features_in + features_out))
        return rng.normal(0, stdev, size=size)

    def __init__(self, *, in_features: int, layers: List[Layer], seed: int):
        layer_sizes = [in_features] + [layer.neurons for layer in layers]
        out_features = layers[-1].neurons

        rng = np.random.Generator(np.random.MT19937(seed))
        self.w = [
            NeuralNetwork.xavier_normal(
                rng, in_features, out_features, size=[dim_out, dim_in])
            for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        self.b = [
            NeuralNetwork.xavier_normal(
                rng, in_features, out_features, size=[layer.neurons, 1])
            for layer in layers
        ]
        self.activations = [layer.activation for layer in layers]
        self.num_layers = len(layers)

        self.x = None
        self.z = [0] * self.num_layers
        self.a = [0] * self.num_layers

        self.dl_da = [0] * self.num_layers
        self.dl_dz = [0] * self.num_layers
        self.dl_dw = [0] * self.num_layers
        self.dl_db = [0] * self.num_layers

    def forward(self, x) -> np.ndarray:
        self.x = x
        self.z[0] = self.w[0] @ x.T + self.b[0]
        self.a[0] = self.g[0].compute(self.z[0])
        for i in range(1, self.n):
            self.z[i] = self.w[i] @ self.a[i-1] + self.b[i]
            self.a[i] = self.g[i].compute(self.z[i])
        return self.a[-1].T

    def backward(self, y_true: np.ndarray, loss: Loss):
        last = self.n - 1
        y_pred = self.a[-1].T
        for i in range(last, -1, -1):
            if i == last:
                self.dl_da[i] = loss.gradient(y_pred, y_true).mean(axis=0).reshape(-1, 1)
            else:
                self.dl_da[i] = self.w[i+1].T @ self.dl_dz[i+1]
            self.dl_dz[i] = self.dl_da[i] * self.g[i].gradient(self.z[i])
            if i == 0:
                self.dl_dw[i] = self.dl_dz[i] @ self.x
            else:
                self.dl_dw[i] = self.dl_dz[i] @ self.a[i-1].T
            self.dl_db[i] = np.mean(self.dl_dz[i], axis=1).reshape(-1, 1)

