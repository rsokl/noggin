from liveplot.xarray import metrics_to_xarrays
from liveplot.typing import LiveMetrics
from liveplot.logger import LiveLogger
from liveplot.plotter import LivePlot

import tests.custom_strategies as cst

import numpy as np
from numpy.testing import assert_array_equal

from hypothesis import given
import pytest


def check_batch_xarray(metrics_dict, metrics_xarray):
    if not metrics_dict:
        assert not metrics_xarray
        return

    # tests for batch-level data
    assert list(metrics_xarray.data_vars) == list(metrics_dict)
    num_iterations = max(len(v["batch_data"]) for v in metrics_dict.values())
    assert_array_equal(
        metrics_xarray.coords["iterations"], np.arange(1, num_iterations + 1)
    )

    for name, data in metrics_dict.items():
        assert_array_equal(
            getattr(metrics_xarray, name),
            data["batch_data"],
            err_msg="(batch) {name} data does not match between the "
            "xarray and the original metric".format(name=name),
        )


def check_epoch_xarray(metrics_dict, metrics_xarray):
    if not metrics_dict:
        assert not metrics_xarray
        return

    # tests for epoch-level data
    assert list(metrics_xarray.data_vars) == list(metrics_dict)

    for name, data in metrics_dict.items():
        xray = getattr(metrics_xarray, name)
        nan_mask = np.logical_not(np.isnan(xray))
        xray_epoch_domain = metrics_xarray.coords["iterations"][nan_mask]
        assert_array_equal(xray_epoch_domain, metrics_dict[name]["epoch_domain"])

        assert_array_equal(
            xray[nan_mask],
            data["epoch_data"],
            err_msg="(epoch) {name} data does not match between the "
            "xarray and the original metric".format(name=name),
        )


@given(metrics=cst.live_metrics())
def test_metrics_to_xarrays(metrics: LiveMetrics):
    batch_xr, epoch_xr = metrics_to_xarrays(metrics)
    check_batch_xarray(metrics_dict=metrics, metrics_xarray=batch_xr)
    check_epoch_xarray(metrics_dict=metrics, metrics_xarray=epoch_xr)


@given(logger=cst.loggers())
def test_logger_xarray(logger: LiveLogger):
    tr_batch, tr_epoch = logger.to_xarray("train")
    te_batch, te_epoch = logger.to_xarray("test")

    check_batch_xarray(logger.train_metrics, tr_batch)
    check_epoch_xarray(logger.train_metrics, tr_epoch)
    check_batch_xarray(logger.test_metrics, te_batch)
    check_epoch_xarray(logger.test_metrics, te_epoch)


@given(logger=cst.loggers())
def test_logger_xarray_validate_inputs(logger: LiveLogger):
    with pytest.raises(ValueError):
        logger.to_xarray("traintest")


@given(plotter=cst.plotters())
def test_plotter_xarray(plotter: LivePlot):
    tr_batch, tr_epoch = plotter.to_xarray("train")
    te_batch, te_epoch = plotter.to_xarray("test")

    check_batch_xarray(plotter.train_metrics, tr_batch)
    check_epoch_xarray(plotter.train_metrics, tr_epoch)
    check_batch_xarray(plotter.test_metrics, te_batch)
    check_epoch_xarray(plotter.test_metrics, te_epoch)
