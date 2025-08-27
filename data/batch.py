from typing import List, Tuple
import numpy as np

class BatchLoader:
    def __init__(self, *, seed: int):
        self.rng = np.random.Generator(np.random.MT19937(seed))
    def load_batch(
            self,
            x: np.ndarray,
            y: np.ndarray,
            *,
            batch_size: int,
            shuffle: bool) -> List[Tuple[np.ndarray, np.ndarray]]:
        if shuffle:
            index_mapping = np.arange(x.shape[0])
            self.rng.shuffle(index_mapping)
            x = x.copy()[index_mapping]
            y = y.copy()[index_mapping]
        return [
            (x[batch_num : batch_num + batch_size,:], y[batch_num : batch_num + batch_size,:])
            for batch_num in range(0, len(x), batch_size)
        ]
