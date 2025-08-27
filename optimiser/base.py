from abc import ABC, abstractmethod

from model import BackwardErrorCache, NeuralNetwork

class Optimiser(ABC):
    @abstractmethod
    def step(self):
        pass
