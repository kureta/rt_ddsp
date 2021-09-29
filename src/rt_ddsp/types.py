from typing import Dict, Union

from numpy import ndarray
from torch import Tensor

TensorDict = Dict[str, Tensor]
AnyTensor = Union[Tensor, ndarray]
