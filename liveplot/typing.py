from numpy import ndarray

from typing import Dict, Union, Sequence
from numbers import Real

ValidColor = Union[str, Real, Sequence[Real], None]
Metrics = Union[
    str, Sequence[str], Dict[str, ValidColor], Dict[str, Dict[str, ValidColor]]
]
LiveMetrics = Dict[str, Dict[str, ndarray]]
