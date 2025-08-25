from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

ParamSet = List[np.ndarray]
AllParams = Tuple[ParamSet, ParamSet]

@dataclass
class BackwardErrorCache:
    weights_error: ParamSet
    biases_error: ParamSet

