from liveplot.xarray import metrics_to_xarrays
from liveplot.typing import LiveMetrics

import tests.custom_strategies as cst

import numpy as np
from numpy.testing import assert_array_equal
from hypothesis import given


@given(metrics=cst.metrics())
def test_metrics_to_xarrays(metrics: LiveMetrics):
    batch_xr, epoch_xr = metrics_to_xarrays(metrics)
    if not metrics:
        assert not batch_xr
        assert not epoch_xr
        return

    # tests for batch-level data
    assert list(batch_xr.data_vars) == list(metrics)
    num_iterations = max(len(v["batch_data"]) for v in metrics.values())
    assert_array_equal(batch_xr.coords["iterations"], np.arange(1, num_iterations + 1))

    for name, data in metrics.items():
        assert_array_equal(getattr(batch_xr, name), data["batch_data"])

    # tests for batch-level data
    assert list(epoch_xr.data_vars) == list(metrics)
    key = list(metrics.keys())[0]
    epoch_iterations = metrics[key]["epoch_domain"]
    assert_array_equal(epoch_xr.coords["iterations"], epoch_iterations)

    for name, data in metrics.items():
        assert_array_equal(getattr(epoch_xr, name), data["epoch_data"])

