from math import isclose
from time import sleep, time
from typing import Optional
import os

import hypothesis.strategies as st
import numpy as np
import pytest
import tests.custom_strategies as cst
from hypothesis import given, settings
from numpy.testing import assert_array_equal
from tests import close_plots

from noggin import LivePlot


class ControlledPlot(LivePlot):
    """Mocks out plotting. Forces plotting to take a specified
    time; no plots are created"""

    def __init__(self, *args, plot_time, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_time = plot_time
        self._liveplot = True

    def _init_plot_window(self):
        return

    def plot(self, plot_batches=True):
        sleep(self.plot_time)


@pytest.mark.skipif(
    os.getenv("HYPOTHESIS_PROFILE") == "ci",
    reason="timing is inconsistent on CI platforms",
)
@pytest.mark.parametrize("plot_time", [0.0001, 0.001, 0.01])
@pytest.mark.parametrize("outer_time", [0.0001, 0.001])
@pytest.mark.parametrize("max_fraction", [0.0, 0.01, 0.2, 0.4])
def test_adaptive_plotting(plot_time, outer_time, max_fraction):
    plotter = ControlledPlot(metrics="a", plot_time=plot_time)

    plotter.max_fraction_spent_plotting = max_fraction
    plotter._queue_size = 1
    total_plot_time = 0.0

    start_time = time()
    for n in range(500):
        if n == 250:
            # bump up plot time to ensure that plot rate
            # adapts appropriately
            plotter.plot_time = 10 * plot_time
        start_plot = time()
        plotter.set_train_batch(dict(a=1), batch_size=1, plot=True)
        total_plot_time += time() - start_plot
        sleep(outer_time)
    total_time = time() - start_time

    actual_fraction = total_plot_time / total_time

    assert isclose(actual_fraction, max_fraction, rel_tol=0.3, abs_tol=0.01)


@pytest.mark.skipif(
    os.getenv("HYPOTHESIS_PROFILE") == "ci",
    reason="timing is inconsistent on CI platforms",
)
@pytest.mark.parametrize(
    ("plot_time", "outer_time", "max_fraction", "expected_fraction"),
    [
        (0.1, 0.001, 1.0, 1.0),
        (0.1, 0.001, 0.7, 0.7),
        (0.1, 0.001, 0.5, 0.5),
        (0.001, 0.001, 1.0, 0.5),
    ],
)
def test_exhaustive_plotting(plot_time, outer_time, max_fraction, expected_fraction):
    """Ensure that plotting dominates runtime when `max_fraction_spent_plotting` is 1
    and plotting is dominant.

    If plotting and outer-loop time are comparable, the fraction of time spent
    plotting should be ~0.5"""

    plotter = ControlledPlot(metrics="a", plot_time=plot_time)
    plotter._max_fraction_spent_plotting = max_fraction
    total_plot_time = 0.0

    start_time = time()
    for n in range(50):
        # continue
        start_plot = time()
        plotter.set_train_batch(dict(a=1), batch_size=1, plot=True)
        total_plot_time += time() - start_plot
        sleep(outer_time)
    total_time = time() - start_time

    actual_fraction = total_plot_time / total_time

    assert isclose(actual_fraction, expected_fraction, rel_tol=0.1, abs_tol=0.01)


@settings(deadline=None, max_examples=20)
@given(plotter=cst.plotters(), liveplot=st.booleans(), plot_batches=st.booleans())
def test_fuzz_plot_method(plotter: LivePlot, liveplot: bool, plot_batches: bool):
    with close_plots():
        plotter._liveplot = liveplot
        plotter.plot(plot_batches=plot_batches)


@settings(deadline=None)
@given(plotter=cst.plotters(), bad_input=cst.everything_except(bool))
def test_validate_plot_input(plotter: LivePlot, bad_input):
    with pytest.raises(TypeError):
        plotter.plot(plot_batches=bad_input)


@settings(deadline=None, max_examples=10)
@given(plotter=cst.plotters(), plot_batches=st.booleans())
def test_plot_batches_flag_via_plot(plotter: LivePlot, plot_batches: bool):
    with close_plots():
        plotter.last_n_batches = None
        plotter.plot(plot_batches=plot_batches)
        for name, metric in plotter._train_metrics.items():
            if metric.batch_domain.size:
                assert bool(metric.batch_line.get_xdata().size) is plot_batches
            if metric.epoch_domain.size:
                assert metric.epoch_line.get_xdata().size > 0

        for name, metric in plotter._test_metrics.items():
            if metric.epoch_domain.size:
                assert metric.epoch_line.get_xdata().size > 0


def test_subsequent_plot_batches_is_false_clears_batch_plot():
    with close_plots():
        plotter = LivePlot("a")
        plotter.set_train_batch(dict(a=1), batch_size=1, plot=True)
        plotter.plot(plot_batches=True)
        assert plotter._train_metrics["a"].batch_line.get_xdata().size > 0
        plotter.plot(plot_batches=False)
        assert plotter._train_metrics["a"].batch_line.get_xdata().size == 0


@settings(deadline=None, max_examples=10)
@given(plotter=cst.plotters(), plot_batches=st.booleans())
def test_plot_batches_flag_via_set_batch(plotter: LivePlot, plot_batches: bool):
    with close_plots():
        plotter.last_n_batches = None
        plotter._liveplot = True
        plotter.max_fraction_spent_plotting = 1.0
        plotter.set_train_batch({}, batch_size=1, plot=plot_batches)
        plotter.plot_train_epoch()
        plotter.plot_test_epoch()
        for name, metric in plotter._train_metrics.items():
            if metric.batch_domain.size:
                assert metric.batch_line is None or (
                    bool(metric.batch_line.get_xdata().size) is plot_batches
                )
            if metric.epoch_domain.size:
                assert metric.epoch_line.get_xdata().size > 0

        for name, metric in plotter._test_metrics.items():
            if metric.epoch_domain.size:
                assert metric.epoch_line.get_xdata().size > 0


@settings(deadline=None, max_examples=10)
@given(
    last_n_batches=st.none() | st.integers(1, 100),
    train_data=st.lists(st.floats(-1e6, 1e6), min_size=1).map(np.array),
    data=st.data(),
)
def test_plot_last_n_batches(
    last_n_batches: Optional[int], train_data: np.ndarray, data: st.DataObject
):
    """Ensures correctness of line-data for varying 'last-n-batches' settings"""
    section = slice(None) if last_n_batches is None else slice(-last_n_batches, None)

    with close_plots():
        epochs = data.draw(
            st.lists(st.sampled_from(range(1, len(train_data) + 1)), unique=True).map(
                np.sort
            ),
            label="epochs",
        )
        plotter = LivePlot(
            "a", last_n_batches=last_n_batches, max_fraction_spent_plotting=1.0
        )
        plotter._liveplot = True
        for n, datum in enumerate(train_data):
            plotter.set_train_batch(dict(a=datum), batch_size=1, plot=True)
            if n + 1 in epochs:
                plotter.plot_train_epoch()

            # check batches
            actual_batchx = plotter._train_metrics["a"].batch_line.get_xdata()
            actual_batchy = plotter._train_metrics["a"].batch_line.get_ydata()
            expected_batchx = np.arange(1, len(train_data) + 1)[: n + 1][section]
            expected_batchy = train_data[: n + 1][section]

            assert_array_equal(actual_batchx, expected_batchx)
            assert_array_equal(actual_batchy, expected_batchy)

            # check epochs
            actual_epochx = plotter._train_metrics["a"].epoch_line.get_xdata()
            actual_epochy = plotter._train_metrics["a"].epoch_line.get_ydata()
            mask = np.logical_and(
                expected_batchx.min() <= plotter.train_metrics["a"]["epoch_domain"],
                plotter.train_metrics["a"]["epoch_domain"] <= expected_batchx.max(),
            )
            expected_epochx = plotter.train_metrics["a"]["epoch_domain"][mask]
            expected_epochy = plotter.train_metrics["a"]["epoch_data"][mask]
            assert_array_equal(actual_epochx, expected_epochx)
            assert_array_equal(actual_epochy, expected_epochy)
