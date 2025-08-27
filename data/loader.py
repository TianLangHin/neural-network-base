from sklearn.datasets import load_breast_cancer
from typing import Tuple
import numpy as np

PartitionedData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    cancer_data = load_breast_cancer()
    x = cancer_data.data
    y = 1 - cancer_data.target.reshape(-1, 1)
    return x, y

def train_test_split(
        x: np.ndarray, y: np.ndarray, *, train_ratio: float, seed: int) -> PartitionedData:
    rng = np.random.Generator(np.random.MT19937(seed))
    index_mapping = np.arange(x.shape[0])
    rng.shuffle(index_mapping)
    shuffled_x = x.copy()[index_mapping]
    shuffled_y = y.copy()[index_mapping]
    boundary = int(x.shape[0] * train_ratio)
    split_x = shuffled_x[:boundary], shuffled_x[boundary:]
    split_y = shuffled_y[:boundary], shuffled_y[boundary:]
    return *split_x, *split_y
