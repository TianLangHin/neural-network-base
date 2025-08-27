from abc import ABC, abstractmethod
import numpy as np

class DataScaler(ABC):
    '''
    Abstract base class for all scaling preprocessing resources to inherit from.
    '''
    @abstractmethod
    def fit(self, x: np.ndarray):
        pass
    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        pass

class ZScoreScaler(DataScaler):
    def __init__(self, features: int):
        self.features = features
        self.mu = [0] * features
        self.std = [0] * features
    def fit(self, x: np.ndarray):
        for i in range(self.features):
            self.mu[i] = x[:,i].mean()
            self.std[i] = x[:,i].std()
    def transform(self, x: np.ndarray) -> np.ndarray:
        for i in range(self.features):
            x[:,i] = (x[:,i] - self.mu[i]) / self.std[i]
        return x
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

class MinMaxScaler(DataScaler):
    def __init__(self, features: int, mapped_min: int | float, mapped_max: int | float):
        self.features = features
        self.min_prev = [0] * features
        self.max_prev = [0] * features
        self.min_new = mapped_min
        self.max_new = mapped_max
    def fit(self, x: np.ndarray):
        for i in range(self.features):
            self.min_prev[i] = x[:,i].min()
            self.max_prev[i] = x[:,i].max()
    def transform(self, x: np.ndarray) -> np.ndarray:
        coefficient = self.max_new - self.min_new
        for i in range(self.features):
            constant = self.min_prev[i] * self.max_new - self.max_prev[i] * self.min_new
            denominator = self.max_prev[i] - self.min_prev[i]
            x[:,i] = (coefficient * x[:,i] + constant) / denominator
        return x
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

