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
from liveplot.logger import LiveLogger
from xarray import Dataset

from typing import Dict, Tuple, Union

LiveObject = Union[LivePlot, LiveLogger]

__all__ = ['metrics_to_DataArrays', 'get_DataArrays', 'concat_experiments']


def metrics_to_DataArrays(metrics: Dict[str, Dict[str, ndarray]]) -> Tuple[Dataset, Dataset]:
    """
    Given liveplot metrics, returns xarray datasets for the batch-level and epoch-level
    metrics, respectively.

    Parameters
    ----------
    metrics : Dict[str, Dict[str, ndarray]]
        Live metrics reported as a dictionary, (e.g. via `LivePlot.train_metrics`
        or `LivePlot.test_metrics`)

    Returns
    -------
    Tuple[xarray.Dataset, xarray.Dataset]
        The batch-level and epoch-level datasets. The metrics are reported as 
        data variables in the dataset, and the coordinates corresponds to
        the batch-iteration count.

    Notes
    -----
    The layout of the resulting data sets are:

    Dimensions:     (iterations: num_iterations)
    Coordinates:
      * iterations  (iterations) int64 1 2 3 ...
    Data variables:
        metric0      (iterations) float64 val_0 val_1 ...
        metric1      (iterations) float64 val_0 val_1 ...
        ...
    """
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


def get_DataArrays(logger: LiveObject) -> Dict[str, Dict[str, Dataset]]:
    """
    Given a LiveLogger or LivePlot instance, returns xarray data sets for
    its train & test, batch-level & epoch-level metrics, respectively.

    Parameters
    ----------
    logger : Union[LivePlot, LiveLogger]

    Returns
    -------
    Dict[str, Dict[str, Dataset]]
        'train/test' -> 'batch/epoch' -> Dataset

    Notes
    -----
    The layout of the resulting data sets are:

    Dimensions:     (iterations: num_iterations)
    Coordinates:
      * iterations  (iterations) int64 1 2 3 ...
    Data variables:
        metric0      (iterations) float64 val_0 val_1 ...
        metric1      (iterations) float64 val_0 val_1 ...
        ...
    """
    tr_batch, tr_epoch = metrics_to_DataArrays(logger.train_metrics)
    te_batch, te_epoch = metrics_to_DataArrays(logger.test_metrics)
    return dict(train=dict(batch=tr_batch, epoch=tr_epoch),
                test=dict(batch=te_batch, epoch=te_epoch))


def concat_experiments(*exps: Dataset) -> Dataset:
    """
    Concatenates xarray data sets from a sequence of experiments.

    Specifically, data sets that record identical metrics measured
    across several independent experiments will be concatenated along
    a new dimension, 'experiment', which tracks the experiment-index
    associated with the corresponding array of metrics.

    Parameters
    ----------
    *exps: Dataset
        One or more data sets recording metrics across independent
        runs of an experiment.

    Returns
    -------
    Dataset
        The recorded metrics joined into a single data set, along an
        experiment-index dimension.

    Notes
    -----
    The form of the resulting Dataset is:

    Dimensions:     (experiment: num_exps, iterations: max_num_its)
    Coordinates:
      * experiment  (experiment) int32 0 1 2 ...
      * iterations  (iterations) int64 1 2 3 ...
    Data variables:
        metric0      (experiment, iterations) float64 val_0 val_1 ...
        metric1      (experiment, iterations) float64 val_0 val_1 ...
        ...
    """
    exp_inds = list(range(len(exps)))
    exp_coord = xr.DataArray(exp_inds, 
                             name='experiment', 
                             dims=['experiment'], 
                             coords=[exp_inds])
    return xr.concat(exps, exp_coord)
