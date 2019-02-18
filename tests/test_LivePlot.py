from tests import err_msg

from liveplot.logger import LiveLogger
from liveplot.plotter import LivePlot

import pytest

from numpy.testing import assert_array_equal

from hypothesis import settings, assume
import hypothesis.strategies as st
from hypothesis.strategies import SearchStrategy
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule, precondition
import numpy as np

def test_redundant_metrics():
    with pytest.raises(ValueError):
        LivePlot(["a", "b", "a"])

    with pytest.raises(ValueError):
        LivePlot(["a", "b", "a", "c", "c"])


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
    """Ensures that LivePlot and LiveLogger log metrics information
    identically"""
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
        self.train_metric_names = ["a", "b", "c"][:num_train_metrics]

        self.test_metric_names = ["a", "b", "c"][:num_test_metrics]
        self.plotter = LivePlot(sorted(set(self.train_metric_names + self.test_metric_names)), refresh=-1)
        self.logger = LiveLogger()

    @rule()
    def get_repr(self):
        """ Ensure no side effect """
        repr(self.logger)
        repr(self.plotter)

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


TestLivePlot = LivePlotStateMachine.TestCase