from loss.base import Loss
import numpy as np

class BinaryCrossEntropy(Loss):
    '''
    Represents the implementation of the binary cross-entropy function.

    To ensure numerical stability, the predicted probabilities are clipped
    to a range of (`eps`, `1-eps`) where `eps` is a small number.
    '''
    def __init__(self, *, eps=1e-8):
        self.eps = eps
    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        '''
        First clips `y_pred` to (`eps`, `1-eps`) before computing the binary cross-entropy.
        '''
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        '''
        Computes, at a given value of `y_pred` and `y_true` with respect to `y_pred`.
        '''
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
