"""
This module provides functionality for converting liveplot metrics to xarray objects,
and for building a dataset from multiple iterations of an experiment.
"""

try:
    import xarray as xr
except ImportError:
    raise ImportError("The Python package `xarray` must be installed "
                      "in order to access this functionality in liveplot.")

import numpy as np

from numpy import ndarray
from liveplot.plotter import LivePlot
from xarray import Dataset

from typing import Dict, Tuple


def metrics_to_DataArrays(metrics: Dict[str, Dict[str, ndarray]]) -> Tuple[Dataset, Dataset]:
    batch_arrays = []
    for metric_name in metrics.keys():
        dat = metrics[metric_name]['batch_data']
        at = xr.DataArray(dat,
                          dims=('iterations',),
                          coords=[np.arange(1, len(dat) + 1)],
                          name=metric_name)
        batch_arrays.append(at)

    epoch_arrays = []
    for metric_name in metrics.keys():
        dat = metrics[metric_name]['epoch_data']
        at = xr.DataArray(dat,
                          dims=('iterations',),
                          coords=[metrics[metric_name]['epoch_domain'].astype(np.int32)],
                          name=metric_name)
        epoch_arrays.append(at)

    return xr.merge(batch_arrays), xr.merge(epoch_arrays)


def plotter_to_DataArrays(plotter: LivePlot) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    pass


def concat_experiments(*exps):
    exp_inds = list(range(len(exps)))
    exp_coord = xr.DataArray(exp_inds, 
                             name='experiment', 
                             dims=['experiment'], 
                             coords=[exp_inds])
    return xr.concat(exps, exp_coord)
