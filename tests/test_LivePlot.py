from tests import err_msg

from liveplot import save_metrics, load_metrics
from liveplot.logger import LiveLogger
from liveplot.plotter import LivePlot

from numpy.testing import assert_array_equal

from hypothesis import settings, assume, note
import hypothesis.strategies as st
from hypothesis.strategies import SearchStrategy
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule, precondition

import numpy as np
from matplotlib.pyplot import Figure, Axes

from contextlib import contextmanager


import pytest


@contextmanager
def close_fig(fig):
    from matplotlib.pyplot import close
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


def test_plot_grid():
    """Ensure that axes have the right type/shape for a given grid spec"""
    fig, ax = LivePlot(["metric a"]).plot_objects()
    with close_fig(fig):
        assert isinstance(ax, Axes)

    fig, ax = LivePlot(["metric a"], nrows=1).plot_objects()
    with close_fig(fig):
        assert isinstance(ax, Axes)

    fig, ax = LivePlot(["metric a"], ncols=1).plot_objects()
    with close_fig(fig):
        assert isinstance(ax, Axes)

    fig, ax = LivePlot(["metric a", "metric_b", "metric_c"], nrows=2, ncols=2).plot_objects()
    with close_fig(fig):
        assert isinstance(ax, np.ndarray)
        assert ax.shape == (2, 2)

    fig, ax = LivePlot(["metric a", "metric_b", "metric_c"]).plot_objects()
    with close_fig(fig):
        assert isinstance(ax, np.ndarray)
        assert ax.shape == (3,)

    fig, ax = LivePlot(["metric a", "metric_b", "metric_c"], ncols=3).plot_objects()
    with close_fig(fig):
        assert isinstance(ax, np.ndarray)
        assert ax.shape == (3,)


def test_trivial_case():
    """ Perform a trivial sanity check on live logger"""
    plotter = LivePlot("a", refresh=-1)
    plotter.set_train_batch(dict(a=1.), batch_size=1, plot=False)
    plotter.set_train_batch(dict(a=3.), batch_size=1, plot=False)
    plotter.plot_train_epoch()

    assert_array_equal(plotter.train_metrics['a']['batch_data'], np.array([1., 3.]))
    assert_array_equal(plotter.train_metrics['a']['epoch_domain'], np.array([2]))
    assert_array_equal(plotter.train_metrics['a']['epoch_data'], np.array([1. / 2. + 3. / 2.]))


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

    @initialize(num_train_metrics=st.integers(0, 3),
                num_test_metrics=st.integers(0, 3))
    def choose_metrics(self,
                       num_train_metrics: int,
                       num_test_metrics: int):
        assume(num_train_metrics + num_test_metrics > 0)
        self.train_metric_names = ["metric-a", "metric-b", "metric-c"][:num_train_metrics]

        self.test_metric_names = ["metric-a", "metric-b", "metric-c"][:num_test_metrics]
        self.plotter = LivePlot(sorted(set(self.train_metric_names + self.test_metric_names)), refresh=-1)
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
            assert isinstance(ax, np.ndarray) and all(isinstance(a, Axes) for a in ax.flat), \
                "An array of axes is expected when multiple metrics are specified."
        else:
            assert isinstance(ax, Axes), "A sole `Axes` instance is expected as the plot " \
                                         "object when only one metric is specified"

    @rule(batch_size=st.integers(0, 2), data=st.data(), plot=st.booleans())
    def set_train_batch(self, batch_size: int, data: SearchStrategy, plot: bool):
        self.train_batch_set = True

        batch = {name: data.draw(st.floats(-1, 1), label=name)
                 for name in self.train_metric_names}
        self.logger.set_train_batch(metrics=batch, batch_size=batch_size)
        self.plotter.set_train_batch(metrics=batch, batch_size=batch_size, plot=plot)

    @rule()
    def set_train_epoch(self):
        self.logger.set_train_epoch()
        self.plotter.plot_train_epoch()

    @rule(batch_size=st.integers(0, 2), data=st.data())
    def set_test_batch(self, batch_size: int, data: SearchStrategy):
        self.test_batch_set = True

        batch = {name: data.draw(st.floats(-1, 1), label=name)
                 for name in self.test_metric_names}
        self.logger.set_test_batch(metrics=batch, batch_size=batch_size)
        self.plotter.set_test_batch(metrics=batch, batch_size=batch_size)

    @rule()
    def set_test_epoch(self):
        self.logger.set_test_epoch()
        self.plotter.plot_test_epoch()

    @precondition(lambda self: self.train_batch_set)
    @rule()
    def compare_train_metrics(self):
        log_metrics = self.logger.train_metrics
        plot_metrics = self.plotter.train_metrics
        
        assert sorted(log_metrics) == sorted(plot_metrics), "The logged train metrics do not match"

        for metric in log_metrics:
            log_batch_data = log_metrics[metric]["batch_data"]
            log_epoch_domain = log_metrics[metric]["epoch_domain"]
            log_epoch_data = log_metrics[metric]["epoch_data"]

            plot_batch_data = plot_metrics[metric]["batch_data"]
            plot_epoch_domain = plot_metrics[metric]["epoch_domain"]
            plot_epoch_data = plot_metrics[metric]["epoch_data"]

            assert_array_equal(log_batch_data, plot_batch_data,
                               err_msg=err_msg(actual=plot_batch_data,
                                               desired=log_batch_data,
                                               name=metric + ": Batch Data"))
            
            assert_array_equal(log_epoch_data, plot_epoch_data,
                               err_msg=err_msg(actual=plot_epoch_data,
                                               desired=log_epoch_data,
                                               name=metric + ": Epoch Data"))

            assert_array_equal(log_epoch_domain, plot_epoch_domain,
                               err_msg=err_msg(actual=plot_epoch_domain,
                                               desired=log_epoch_domain,
                                               name=metric + ": Epoch Domain"))

    @precondition(lambda self: self.test_batch_set)
    @rule()
    def compare_test_metrics(self):
        log_metrics = self.logger.test_metrics
        plot_metrics = self.plotter.test_metrics

        assert sorted(log_metrics) == sorted(plot_metrics), "The logged test metrics do not match"

        for metric in log_metrics:
            log_batch_data = log_metrics[metric]["batch_data"]
            log_epoch_domain = log_metrics[metric]["epoch_domain"]
            log_epoch_data = log_metrics[metric]["epoch_data"]

            plot_batch_data = plot_metrics[metric]["batch_data"]
            plot_epoch_domain = plot_metrics[metric]["epoch_domain"]
            plot_epoch_data = plot_metrics[metric]["epoch_data"]

            assert_array_equal(log_batch_data, plot_batch_data,
                               err_msg=err_msg(actual=plot_batch_data,
                                               desired=log_batch_data,
                                               name=metric + ": Batch Data"))

            assert_array_equal(log_epoch_data, plot_epoch_data,
                               err_msg=err_msg(actual=plot_epoch_data,
                                               desired=log_epoch_data,
                                               name=metric + ": Epoch Data"))

            assert_array_equal(log_epoch_domain, plot_epoch_domain,
                               err_msg=err_msg(actual=plot_epoch_domain,
                                               desired=log_epoch_domain,
                                               name=metric + ": Epoch Domain"))

    @rule(save_via_object=st.booleans())
    def check_metric_io(self, save_via_object: bool):
        """Ensure the saving/loading metrics always produces self-consistent
        results with the plotter"""
        from uuid import uuid4
        filename = str(uuid4())
        if save_via_object:
            save_metrics(filename, liveplot=self.plotter)
        else:
            save_metrics(filename,
                         train_metrics=self.plotter.train_metrics,
                         test_metrics=self.plotter.test_metrics)
        io_train_metrics, io_test_metrics = load_metrics(filename)

        plot_train_metrics = self.plotter.train_metrics
        plot_test_metrics = self.plotter.test_metrics

        assert tuple(io_test_metrics) == tuple(plot_test_metrics), "The io test metrics do not match those " \
                                                                   "from the LivePlot instance. Order matters " \
                                                                   "for reproducing the plot."

        for metric in io_train_metrics:
            io_batch_data = io_train_metrics[metric]["batch_data"]
            io_epoch_domain = io_train_metrics[metric]["epoch_domain"]
            io_epoch_data = io_train_metrics[metric]["epoch_data"]

            plot_batch_data = plot_train_metrics[metric]["batch_data"]
            plot_epoch_domain = plot_train_metrics[metric]["epoch_domain"]
            plot_epoch_data = plot_train_metrics[metric]["epoch_data"]

            assert_array_equal(io_batch_data, plot_batch_data,
                               err_msg=err_msg(actual=plot_batch_data,
                                               desired=io_batch_data,
                                               name=metric + ": Batch Data (train)"))

            assert_array_equal(io_epoch_data, plot_epoch_data,
                               err_msg=err_msg(actual=plot_epoch_data,
                                               desired=io_epoch_data,
                                               name=metric + ": Epoch Data (train)"))

            assert_array_equal(io_epoch_domain, plot_epoch_domain,
                               err_msg=err_msg(actual=plot_epoch_domain,
                                               desired=io_epoch_domain,
                                               name=metric + ": Epoch Domain (train)"))

        for metric in io_test_metrics:
            io_batch_data = io_test_metrics[metric]["batch_data"]
            io_epoch_domain = io_test_metrics[metric]["epoch_domain"]
            io_epoch_data = io_test_metrics[metric]["epoch_data"]

            plot_batch_data = plot_test_metrics[metric]["batch_data"]
            plot_epoch_domain = plot_test_metrics[metric]["epoch_domain"]
            plot_epoch_data = plot_test_metrics[metric]["epoch_data"]

            assert_array_equal(io_batch_data, plot_batch_data,
                               err_msg=err_msg(actual=plot_batch_data,
                                               desired=io_batch_data,
                                               name=metric + ": Batch Data (test)"))

            assert_array_equal(io_epoch_data, plot_epoch_data,
                               err_msg=err_msg(actual=plot_epoch_data,
                                               desired=io_epoch_data,
                                               name=metric + ": Epoch Data (test)"))

            assert_array_equal(io_epoch_domain, plot_epoch_domain,
                               err_msg=err_msg(actual=plot_epoch_domain,
                                               desired=io_epoch_domain,
                                               name=metric + ": Epoch Domain (test)"))

    def teardown(self):
        if hasattr(self, "plotter") and self.plotter._fig is not None:
            from matplotlib.pyplot import close
            close(self.plotter._fig)
        super().teardown()


@pytest.mark.usefixtures("cleandir")
class TestLivePlot(LivePlotStateMachine.TestCase):
    pass
