from loss.base import Loss
import numpy as np

class MeanSquaredError(Loss):
    '''
    Represents the implementation of the mean squared error function.
    '''
    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) * (y_pred - y_true)
    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true)
