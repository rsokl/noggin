from tests.utils import compare_all_metrics

from liveplot import save_metrics, load_metrics
from liveplot.logger import LiveLogger
from liveplot.plotter import LivePlot

import numpy as np
from numpy.testing import assert_array_equal
from numpy import ndarray

from hypothesis import settings, assume, note
import hypothesis.strategies as st
from hypothesis.strategies import SearchStrategy
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    rule,
    precondition,
    invariant,
)

from matplotlib.pyplot import Figure, Axes, close

from contextlib import contextmanager
from string import ascii_letters

import pytest


@contextmanager
def close_fig(fig):
    try:
        yield None
    finally:
        close(fig)


def test_redundant_metrics():
    """Ensures that redundant metrics are not permitted"""
    with pytest.raises(ValueError):
        LivePlot(["a", "b", "a"])

    with pytest.raises(ValueError):
        LivePlot(["a", "b", "a", "c", "c"])


@pytest.mark.parametrize(
    ("num_metrics", "fig_layout", "outer_type", "shape"),
    [
        (1, dict(), Axes, tuple()),
        (1, dict(nrows=1), Axes, tuple()),
        (1, dict(ncols=1), Axes, tuple()),
        (3, dict(nrows=2, ncols=2), ndarray, (2, 2)),
        (3, dict(), ndarray, (3,)),
        (3, dict(nrows=3), ndarray, (3,)),
        (3, dict(ncols=3), ndarray, (3,)),
    ],
)
def test_plot_grid(num_metrics, fig_layout, outer_type, shape):
    """Ensure that axes have the right type/shape for a given grid spec"""
    metric_names = list(ascii_letters[:num_metrics])

    fig, ax = LivePlot(metric_names, **fig_layout).plot_objects()

    assert isinstance(fig, Figure)
    with close_fig(fig):
        assert isinstance(ax, outer_type)
        if shape:
            assert ax.shape == shape


def test_trivial_case():
    """ Perform a trivial sanity check on live logger"""
    plotter = LivePlot("a", refresh=-1)
    plotter.set_train_batch(dict(a=1.0), batch_size=1, plot=False)
    plotter.set_train_batch(dict(a=3.0), batch_size=1, plot=False)
    plotter.plot_train_epoch()

    assert_array_equal(plotter.train_metrics["a"]["batch_data"], np.array([1.0, 3.0]))
    assert_array_equal(plotter.train_metrics["a"]["epoch_domain"], np.array([2]))
    assert_array_equal(
        plotter.train_metrics["a"]["epoch_data"], np.array([1.0 / 2.0 + 3.0 / 2.0])
    )


@settings(deadline=None)
class LivePlotStateMachine(RuleBasedStateMachine):
    """Ensures that:
    - LivePlot and LiveLogger log metrics information identically.
    - Calling methods do not have unintended side-effects
    - Metric IO is self-consistent
    - Plot objects are produced as expected"""

    def __init__(self):
        super().__init__()
        self.train_metric_names = []
        self.test_metric_names = []
        self.train_batch_set = False
        self.test_batch_set = False
        self.plotter = None  # type: LivePlot
        self.logger = None  # type: LiveLogger

    @initialize(num_train_metrics=st.integers(0, 3), num_test_metrics=st.integers(0, 3))
    def choose_metrics(self, num_train_metrics: int, num_test_metrics: int):
        assume(num_train_metrics + num_test_metrics > 0)
        self.train_metric_names = ["metric-a", "metric-b", "metric-c"][
            :num_train_metrics
        ]

        self.test_metric_names = ["metric-a", "metric-b", "metric-c"][:num_test_metrics]
        self.plotter = LivePlot(
            sorted(set(self.train_metric_names + self.test_metric_names)), refresh=-1
        )
        self.logger = LiveLogger()

        note("Train metric names: {}".format(self.train_metric_names))
        note("Test metric names: {}".format(self.test_metric_names))

    @rule()
    def get_repr(self):
        """ Ensure no side effect """
        repr(self.logger)
        repr(self.plotter)

    @rule()
    def check_plt_objects(self):
        """ Ensure no side effect """
        fig, ax = self.plotter.plot_objects()

        assert isinstance(fig, Figure)

        if len(set(self.train_metric_names + self.test_metric_names)) > 1:
            assert isinstance(ax, np.ndarray) and all(
                isinstance(a, Axes) for a in ax.flat
            ), "An array of axes is expected when multiple metrics are specified."
        else:
            assert isinstance(ax, Axes), (
                "A sole `Axes` instance is expected as the plot "
                "object when only one metric is specified"
            )

    @rule(batch_size=st.integers(0, 2), data=st.data(), plot=st.booleans())
    def set_train_batch(self, batch_size: int, data: SearchStrategy, plot: bool):
        self.train_batch_set = True

        batch = {
            name: data.draw(st.floats(-1, 1), label=name)
            for name in self.train_metric_names
        }
        self.logger.set_train_batch(metrics=batch, batch_size=batch_size)
        self.plotter.set_train_batch(metrics=batch, batch_size=batch_size, plot=plot)

    @rule()
    def set_train_epoch(self):
        self.logger.set_train_epoch()
        self.plotter.plot_train_epoch()

    @rule(batch_size=st.integers(0, 2), data=st.data())
    def set_test_batch(self, batch_size: int, data: SearchStrategy):
        self.test_batch_set = True

        batch = {
            name: data.draw(st.floats(-1, 1), label=name)
            for name in self.test_metric_names
        }
        self.logger.set_test_batch(metrics=batch, batch_size=batch_size)
        self.plotter.set_test_batch(metrics=batch, batch_size=batch_size)

    @rule()
    def set_test_epoch(self):
        self.logger.set_test_epoch()
        self.plotter.plot_test_epoch()

    @precondition(lambda self: self.train_batch_set)
    @invariant()
    def compare_train_metrics(self):
        log_metrics = self.logger.train_metrics
        plot_metrics = self.plotter.train_metrics
        compare_all_metrics(log_metrics, plot_metrics)

    @precondition(lambda self: self.test_batch_set)
    @invariant()
    def compare_test_metrics(self):
        log_metrics = self.logger.test_metrics
        plot_metrics = self.plotter.test_metrics
        compare_all_metrics(log_metrics, plot_metrics)

    @rule(save_via_object=st.booleans())
    def check_metric_io(self, save_via_object: bool):
        """Ensure the saving/loading metrics always produces self-consistent
        results with the plotter"""
        from uuid import uuid4

        filename = str(uuid4())
        if save_via_object:
            save_metrics(filename, liveplot=self.plotter)
        else:
            save_metrics(
                filename,
                train_metrics=self.plotter.train_metrics,
                test_metrics=self.plotter.test_metrics,
            )
        io_train_metrics, io_test_metrics = load_metrics(filename)

        plot_train_metrics = self.plotter.train_metrics
        plot_test_metrics = self.plotter.test_metrics

        assert tuple(io_test_metrics) == tuple(plot_test_metrics), (
            "The io test metrics do not match those from the LivePlot "
            "instance. Order matters for reproducing the plot."
        )

        compare_all_metrics(plot_train_metrics, io_train_metrics)
        compare_all_metrics(plot_test_metrics, io_test_metrics)

    @precondition(lambda self: self.plotter is not None)
    @invariant()
    def check_from_dict_roundtrip(self):
        plotter_dict = self.plotter.to_dict()
        new_plotter = LivePlot.from_dict(plotter_dict)

        for attr in [
            "_num_train_epoch",
            "_num_train_batch",
            "_num_test_epoch",
            "_num_test_batch",
            "refresh",
            "_metrics",
            "_pltkwargs",
            "metric_colors",
        ]:
            desired = getattr(self.plotter, attr)
            actual = getattr(new_plotter, attr)
            assert actual == desired, (
                "LiveLogger.from_metrics did not round-trip successfully.\n"
                "logger.{} does not match.\nGot: {}\nExpected: {}"
                "".format(attr, actual, desired)
            )

        compare_all_metrics(self.plotter.train_metrics, new_plotter.train_metrics)
        compare_all_metrics(self.plotter.test_metrics, new_plotter.test_metrics)

    def teardown(self):
        if self.plotter is not None:
            fig, _ = self.plotter.plot_objects()
            if fig is not None:
                close(fig)
        super().teardown()


@pytest.mark.usefixtures("cleandir")
class TestLivePlot(LivePlotStateMachine.TestCase):
    pass
