from activation.base import Activation
import numpy as np

class Sigmoid(Activation):
    '''
    Represents the implementation of the sigmoid activation function.
    '''
    def compute(self, x: np.ndarray) -> np.ndarray:
        '''
        Due to the presence of the exponent term in `sigma(x) = 1 / (1 + e^(-x))`,
        numerical instability can occur if `x` is a very negative value.

        We obtain an alternative representation by multiplying both the
        numerator and denominator by `e^x`, which will be a low value if `x` is negative.
        It follows that `sigma(x) = e^x / (1 + e^x)`.

        Hence, when `x >= 0`, we use `sigma(x) = 1 / (1 + e^(-x))`, and
        when `x < 0`, `sigma(x) = e^x / (1 + e^x)`.
        '''
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    def gradient(self, x: np.ndarray) -> np.ndarray:
        '''
        The derivative of the sigmoid function is computable in terms of
        the actual function value at that point.

        Let this value be `f(x)`. Its derivative is `f(x) * (1 - f(x))`.
        '''
        return self.compute(x) * (1 - self.compute(x))
