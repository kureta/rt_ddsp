from typing import Any, Dict, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

TensorDict = Dict[str, Tensor]
AnyTensor = Union[Tensor, ndarray]


def is_any_tensor(x: Any):
    return isinstance(x, torch.Tensor) or isinstance(x, np.ndarray)
