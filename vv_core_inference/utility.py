import re
from typing import Any, Union

import numpy
import torch
from torch import Tensor

OPSET = 9


def extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def remove_weight_norm(m):
    try:
        torch.nn.utils.remove_weight_norm(m)
    except ValueError:
        pass


def to_tensor(array: Union[Tensor, numpy.ndarray, Any], device):
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        array = torch.from_numpy(array)
    return array.to(device)
