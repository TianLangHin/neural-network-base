from abc import ABC, abstractmethod

from model import NeuralNetwork
from utils import BackwardErrorCache

class Optimiser(ABC):
    @abstractmethod
    def step(self):
        pass
