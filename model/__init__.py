from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from loss import Loss

Layer = namedtuple('Layer', ['neurons', 'activation'])
ParamSet = List[np.ndarray]
AllParams = Tuple[ParamSet, ParamSet]

@dataclass
class BackwardErrorCache:
    weights_error: ParamSet
    biases_error: ParamSet

class NeuralNetwork:
    def __init__(self, *, in_features: int, layers: List[Layer], seed: int):
        layer_sizes = [in_features] + [layer.neurons for layer in layers]
        out_features = layers[-1].neurons

        rng = np.random.Generator(np.random.MT19937(seed))
        self.w = [
            xavier_normal(
                rng, in_features, out_features, size=[dim_out, dim_in])
            for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        self.b = [
            xavier_normal(
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

        self.error_cache = BackwardErrorCache(
            weights_error=[0] * self.num_layers,
            biases_error=[0] * self.num_layers)

    def forward(self, x) -> np.ndarray:
        activation = self.activations[0]
        self.x = x
        self.z[0] = self.w[0] @ x.T + self.b[0]
        self.a[0] = activation.compute(self.z[0])
        for i in range(1, self.num_layers):
            activation = self.activations[i]
            self.z[i] = self.w[i] @ self.a[i-1] + self.b[i]
            self.a[i] = activation.compute(self.z[i])
        return self.a[-1].T

    def backward(self, y_true: np.ndarray, loss: Loss):
        last = self.num_layers - 1
        y_pred = self.a[-1].T
        for i in range(last, -1, -1):
            if i == last:
                self.dl_da[i] = loss.gradient(y_pred, y_true).mean(axis=0).reshape(-1, 1)
            else:
                self.dl_da[i] = self.w[i+1].T @ self.dl_dz[i+1]
            activation = self.activations[i]
            self.dl_dz[i] = self.dl_da[i] * activation.gradient(self.z[i])
            if i == 0:
                self.error_cache.weights_error[i] = self.dl_dz[i] @ self.x
            else:
                self.error_cache.weights_error[i] = self.dl_dz[i] @ self.a[i-1].T
            self.error_cache.biases_error[i] = np.mean(self.dl_dz[i], axis=1).reshape(-1, 1)

    def update(self, i: int, weight_update: List[np.ndarray], bias_update: List[np.ndarray]):
        self.w[i] -= weight_update
        self.b[i] -= bias_update

    def get_num_layers(self) -> int:
        return self.num_layers

    def get_error_cache(self) -> BackwardErrorCache:
        return self.error_cache

def xavier_normal(
        rng: np.random.Generator,
        features_in: int,
        features_out: int,
        *,
        size: List[int]) -> np.ndarray:
    stdev = np.sqrt(2 / (features_in + features_out))
    return rng.normal(0, stdev, size=size)
