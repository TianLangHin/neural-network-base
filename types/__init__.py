from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

Layer = namedtuple('Layer', ['neurons', 'activation'])
ParamSet = List[np.ndarray]
AllParams = Tuple[ParamSet, ParamSet]

@dataclass
class BackwardErrorCache:
    weights_error: ParamSet
    biases_error: ParamSet


