"""
This module provides functionality for converting noggin metrics to xarray objects,
and for building a dataset from multiple iterations of an experiment.
"""

try:
    import xarray as xr
except ImportError:  # pragma:nocover
    raise ImportError(
        "The Python package `xarray` must be installed "
        "in order to access this functionality in noggin."
    )
from collections import namedtuple
from typing import Dict, Tuple, Union

import numpy as np
from numpy import ndarray
from xarray import Dataset

from noggin.logger import LiveLogger
from noggin.plotter import LivePlot

LiveObject = Union[LivePlot, LiveLogger]

__all__ = ["metrics_to_xarrays", "concat_experiments"]


MetricArrays = namedtuple("MetricArrays", ("batch", "epoch"))


def metrics_to_xarrays(
    metrics: Dict[str, Dict[str, ndarray]]
) -> Tuple[Dataset, Dataset]:
    """
    Given noggin metrics, returns xarray datasets for the batch-level and epoch-level
    metrics, respectively.

    Parameters
    ----------
    metrics : Dict[str, Dict[str, ndarray]]
        Live metrics reported as a dictionary, (e.g. via `LivePlot.train_metrics`
        or `LivePlot.test_metrics`)

    Returns
    -------
    MetricArrays[xarray.Dataset, xarray.Dataset]
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
        dat = metrics[metric_name]["batch_data"]
        at = xr.DataArray(
            dat,
            dims=("iterations",),
            coords=[np.arange(1, len(dat) + 1)],
            name=metric_name,
        )
        batch_arrays.append(at)

    epoch_arrays = []
    for metric_name in metrics.keys():
        dat = metrics[metric_name]["epoch_data"]
        at = xr.DataArray(
            dat,
            dims=("iterations",),
            coords=[metrics[metric_name]["epoch_domain"].astype(np.int32)],
            name=metric_name,
        )
        epoch_arrays.append(at)

    return MetricArrays(batch=xr.merge(batch_arrays), epoch=xr.merge(epoch_arrays))


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
    if not all(bool(x) for x in exps):
        raise ValueError(
            "An empty dataset was included among the experiments: {}".format(exps)
        )

    if len(exps) == 0:
        raise ValueError("At least one dataset must be provided, got: {}".format(exps))

    if not len(set(tuple(x.data_vars) for x in exps)) == 1:
        raise ValueError(
            "All of the provided datasets must have the "
            "same data variables, got: {}".format(tuple(x.data_vars) for x in exps)
        )

    exp_inds = list(range(len(exps)))
    exp_coord = xr.DataArray(
        exp_inds, name="experiment", dims=["experiment"], coords=[exp_inds]
    )
    return xr.concat(exps, exp_coord)
