import numpy as np

from numpy import ndarray
from liveplot.plotter import LivePlot
from liveplot.logger import LiveLogger

from typing import Dict, Tuple, Union, Sequence
from numbers import Real

ValidColor = Union[str, Real, Sequence[Real], None]
Metrics = Union[str, Sequence[str], Dict[str, ValidColor], Dict[str, Dict[str, ValidColor]]]
LiveMetrics = Dict[str, Dict[str, ndarray]]