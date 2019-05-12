from numbers import Real
from typing import Dict, Sequence, Union

from numpy import ndarray

ValidColor = Union[str, Real, Sequence[Real], None]
Metrics = Union[
    str, Sequence[str], Dict[str, ValidColor], Dict[str, Dict[str, ValidColor]]
]
LiveMetrics = Dict[str, Dict[str, ndarray]]
