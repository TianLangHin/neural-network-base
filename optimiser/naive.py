from model import NeuralNetwork
from optimiser.base import Optimiser
from utils import BackwardErrorCache

class NaiveOptimiser(Optimiser):
    def __init__(self, nn: NeuralNetwork, *, learning_rate: float):
        self.nn = nn
        self.lr = learning_rate
    def step(self):
        errors = self.nn.get_error_cache()
        dl_dw, dl_db = errors.weights_error, errors.biases_error
        for i in range(self.nn.get_num_layers()):
            self.nn.update(i, self.lr * dl_dw[i], self.lr * dl_db[i])
