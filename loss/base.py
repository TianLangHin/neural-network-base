from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    @abstractmethod
    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass
