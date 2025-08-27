import numpy as np

from model import BackwardErrorCache, NeuralNetwork
from optimiser.base import Optimiser

class AdamOptimiser(Optimiser):
    def __init__(
            self,
            nn: NeuralNetwork,
            *,
            learning_rate: float,
            beta1: float = 0.9,
            beta2: float = 0.999):
        self.nn = nn
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.v_dw = [0] * nn.get_num_layers()
        self.v_db = [0] * nn.get_num_layers()
        self.s_dw = [0] * nn.get_num_layers()
        self.s_db = [0] * nn.get_num_layers()
    def step(self):
        errors = self.nn.get_error_cache()
        dl_dw, dl_db = errors.weights_error, errors.biases_error
        for i in range(self.nn.get_num_layers()):
            self.v_dw[i] = (self.beta1 * self.v_dw[i]
                            + (1 - self.beta1) * dl_dw[i])
            self.v_db[i] = (self.beta1 * self.v_db[i]
                            + (1 - self.beta1) * dl_db[i])
            self.s_dw[i] = (self.beta2 * self.s_dw[i]
                            + (1 - self.beta2) * dl_dw[i] * dl_dw[i])
            self.s_db[i] = (self.beta2 * self.s_db[i]
                            + (1 - self.beta2) * dl_db[i] * dl_db[i])
            w_numerator = self.v_dw[i] / (1 - self.beta1)
            w_denominator = 1e-8 + np.sqrt(self.s_dw[i] / (1 - self.beta2))
            b_numerator = self.v_db[i] / (1 - self.beta1)
            b_denominator = 1e-8 + np.sqrt(self.s_db[i] / (1 - self.beta2))
            w_update = w_numerator / w_denominator
            b_update = b_numerator / b_denominator
            self.nn.update(i, self.lr * w_update, self.lr * b_update)
