from numbers import Real
from typing import Dict, Sequence, Union
from matplotlib.pyplot import Axes, Figure

from numpy import ndarray

Axes = Axes
Figure = Figure

ValidColor = Union[str, Real, Sequence[Real], None]
Metrics = Union[
    str, Sequence[str], Dict[str, ValidColor], Dict[str, Dict[str, ValidColor]]
]
LiveMetrics = Dict[str, Dict[str, ndarray]]
