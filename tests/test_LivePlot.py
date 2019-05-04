from tests.base_state_machines import LivePlotStateMachine
from contextlib import contextmanager
from string import ascii_letters

import numpy as np
from numpy.testing import assert_array_equal
from numpy import ndarray

from hypothesis import given
import hypothesis.strategies as st
from hypothesis.stateful import rule, precondition, invariant

from matplotlib.pyplot import Figure, Axes
from matplotlib.pyplot import close

import pytest

from liveplot import save_metrics, load_metrics
from liveplot.plotter import LivePlot

from tests.utils import compare_all_metrics
import tests.custom_strategies as cst


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


@given(refresh=st.floats(min_value=-1, max_value=100, exclude_min=True))
def test_init_refresh(refresh: float):
    plotter = LivePlot("loss", refresh=refresh)
    refresh = 0.001 if 0 <= refresh < 0.001 else refresh
    assert plotter.refresh == refresh


@given(refresh=st.floats(min_value=-1, max_value=100, exclude_min=True))
def test_setter_refresh(refresh: float):
    plotter = LivePlot("loss")
    plotter.refresh = refresh
    refresh = 0.001 if 0 <= refresh < 0.001 else refresh
    assert plotter.refresh == refresh


@given(colors=st.lists(cst.matplotlib_colors(), min_size=1, max_size=4))
def test_flat_color_syntax(colors: list):
    metric_names = ascii_letters[: len(colors)]
    p = LivePlot({n: c for n, c in zip(metric_names, colors)})
    assert p.metric_colors == {n: dict(train=c) for n, c in zip(metric_names, colors)}


class LivePlotStateChecker(LivePlotStateMachine):
    """Ensures that:
    - LivePlot and LiveLogger log metrics information identically.
    - Calling methods do not have unintended side-effects
    - Metric IO is self-consistent
    - Plot objects are produced as expected

    Note that this inherits from the base rule-based state machine for
    `LivePlot`"""

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
            assert_array_equal(
                actual,
                desired,
                err_msg="LiveLogger.from_metrics did not round-trip successfully.\n"
                "logger.{} does not match.\nGot: {}\nExpected: {}"
                "".format(attr, actual, desired),
            )

        compare_all_metrics(self.plotter.train_metrics, new_plotter.train_metrics)
        compare_all_metrics(self.plotter.test_metrics, new_plotter.test_metrics)

        assert isinstance(new_plotter._test_colors, type(self.plotter._test_colors))
        assert self.plotter._test_colors == new_plotter._test_colors
        assert self.plotter._test_colors[None] is new_plotter._test_colors[None]

        assert isinstance(self.plotter._train_colors, type(new_plotter._train_colors))
        assert self.plotter._train_colors == new_plotter._train_colors
        assert self.plotter._train_colors[None] is new_plotter._train_colors[None]


@pytest.mark.usefixtures("cleandir")
class TestLivePlot(LivePlotStateChecker.TestCase):
    pass
