from activation.base import Activation
import numpy as np

class ReLU(Activation):
    '''
    Represents the implementation of the ReLU (rectified linear unit) activation function.

    Although this function is not smooth at `x = 0`,
    we define its gradient at that point to be `0` to accommodate for backpropagation
    when it is used as an activation function in a neural network.
    '''
    def compute(self, x: np.ndarray) -> np.ndarray:
        '''
        For negative `x` (`x < 0`), `ReLU(x) = 0`.
        For non-negative `x` (`x >= 0`), `ReLU(x) = x`.

        It follows that `ReLU(x) = max(0, x)`.
        '''
        return np.maximum(0, x)
    def gradient(self, x: np.ndarray) -> np.ndarray:
        '''
        For negative `x` (`x < 0`), `ReLU(x) = 0` and thus `ReLU'(x) = 0`.
        For positive `x` (`x > 0`), `ReLU(x) = x` and thus `ReLU'(x) = 1`.

        In both of the above cases, it can be written that `ReLU'(x) = sgn(ReLU(x))`.

        For `x == 0`, the derivative is not mathematically defined.
        However, we can define `ReLU'(0) = 0` for computational efficiency,
        since `np.sign(0) == 0`.

        Hence, for any `x`, we can compute the gradient of ReLU in terms of its value at `x`,
        meaning `ReLU'(x) = sgn(ReLU(x)`.
        '''
        return np.sign(self.compute(x))
