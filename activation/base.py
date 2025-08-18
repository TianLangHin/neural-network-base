from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    '''
    Abstract base class for all activation functions to inherit from.

    All activation functions that inherit from this class represent
    an element-wise function that can be applied to an object of type `np.ndarray`
    of any dimension. It defines a `compute` method for its computation
    and `gradient` for the computation of its gradient at a particular input value.
    '''
    @abstractmethod
    def compute(self, x: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass
