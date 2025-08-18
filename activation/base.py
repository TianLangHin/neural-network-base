from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    @abstractmethod
    def compute(self, x: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass
