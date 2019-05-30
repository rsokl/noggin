from numbers import Real
from typing import Dict, Sequence, Union
from matplotlib.pyplot import Axes as _Axes
from matplotlib.pyplot import Figure as _Figure

from numpy import ndarray

Axes = _Axes
Figure = _Figure

ValidColor = Union[str, Real, Sequence[Real], None]
Metrics = Union[
    str, Sequence[str], Dict[str, ValidColor], Dict[str, Dict[str, ValidColor]]
]
LiveMetrics = Dict[str, Dict[str, ndarray]]
