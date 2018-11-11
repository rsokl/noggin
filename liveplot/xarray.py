"""
This module provides functionality for converting liveplot metrics to xarray objects,
and for building a dataset from multiple iterations of an experiment.
"""

try:
    import xarray as xr
except ImportError:
    raise ImportError("The Python package `xarray` must be installed "/ 
                      "in order to access this functionality in liveplot.")

from liveplot.plotter import LiveMetric, LivePlot
import numpy as np
from typing import Union, Dict, Iterable


def batch_metrics_to_DataArray(metrics: Dict[str, Dict[str, np.ndarray]]) -> xr.DataArray:
    dat_arrays = []
    for metric_name in metrics.keys():
        dat = metrics[metric_name]['batch_data']
        at = xr.DataArray(dat,
                          dims=('iterations',),
                          coords=[np.arange(1, len(dat) + 1)],
                          name=metric_name)
        dat_arrays.append(at)
    return xr.merge(dat_arrays)

def epoch_metrics_to_DataArray(metrics):
    dat_arrays = []
    for metric_name in metrics.keys():
        dat = metrics[metric_name]['epoch_data']
        at = xr.DataArray(dat, 
                          dims=('iterations',), 
                          coords=[metrics[metric_name]['epoch_domain'].astype(np.int32)], 
                          name=metric_name)
        dat_arrays.append(at)
    return xr.merge(dat_arrays)

def concat_experiments(*exps):
    exp_inds = list(range(len(exps)))
    exp_coord = xr.DataArray(exp_inds, 
                             name='experiment', 
                             dims=['experiment'], 
                             coords=[exp_inds])
    return xr.concat(exps, exp_coord)