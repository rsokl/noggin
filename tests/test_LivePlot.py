from tests import err_msg

from liveplot.logger import LiveLogger
from liveplot.plotter import LivePlot

import pytest

from numpy.testing import assert_array_equal, assert_allclose

from hypothesis import settings
import hypothesis.strategies as st
from hypothesis.strategies import SearchStrategy
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule, precondition


def test_redundant_metrics():
    with pytest.raises(ValueError):
        LivePlot(["a", "b", "a"])

    with pytest.raises(ValueError):
        LivePlot(["a", "b", "a", "c", "c"])


@settings(max_examples=500)
class LivePlotStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.train_metric_names = []
        self.test_metric_names = []

    @initialize(num_train_metrics=st.integers(0, 3),
                num_test_metrics=st.integers(0, 3))
    def choose_metrics(self,
                       num_train_metrics: int,
                       num_test_metrics: int):
        self.train_metric_names = ["a", "b", "c"][:num_train_metrics]

        self.test_metric_names = ["a", "b", "c"][:num_test_metrics]
        self.plotter = LivePlot(sorted(set(self.train_metric_names + self.test_metric_names)), refresh=-1)
        self.logger = LiveLogger()

    @rule()
    def get_repr(self):
        """ Ensure no side effect """
        repr(self.logger)

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
        metrics = self.logger.train_metrics
        assert sorted(metrics) == sorted(m.name for m in self.train_metrics), \
            "The logged train metrics do not match"

        for metric in self.train_metrics:
            batch_data = metrics[metric.name]["batch_data"]
            epoch_domain = metrics[metric.name]["epoch_domain"]
            epoch_data = metrics[metric.name]["epoch_data"]

            assert_array_equal(metric.batch_data, batch_data,
                               err_msg=err_msg(actual=batch_data,
                                               desired=metric.batch_data,
                                               name=metric.name + ": Batch Data"))
            assert_array_equal(metric.epoch_domain, epoch_domain,
                               err_msg=err_msg(actual=epoch_domain,
                                               desired=metric.epoch_domain,
                                               name=metric.name + ": Epoch Domain"))

            assert_allclose(actual=epoch_data,
                            desired=metric.epoch_data,
                            err_msg=err_msg(actual=epoch_data,
                                            desired=metric.epoch_data,
                                            name=metric.name + ": Epoch Data"))

    @precondition(lambda self: self.test_batch_set)
    @rule()
    def compare_test_metrics(self):
        metrics = self.logger.test_metrics
        assert sorted(metrics) == sorted(m.name for m in self.test_metrics), \
            "The logged test metrics do not match"

        for metric in self.test_metrics:
            batch_data = metrics[metric.name]["batch_data"]
            epoch_domain = metrics[metric.name]["epoch_domain"]
            epoch_data = metrics[metric.name]["epoch_data"]

            assert_array_equal(metric.batch_data, batch_data,
                               err_msg=err_msg(actual=batch_data,
                                               desired=metric.batch_data,
                                               name=metric.name + ": Batch Data"))
            assert_array_equal(metric.epoch_domain, epoch_domain,
                               err_msg=err_msg(actual=epoch_domain,
                                               desired=metric.epoch_domain,
                                               name=metric.name + ": Epoch Domain"))

            assert_allclose(actual=epoch_data,
                            desired=metric.epoch_data,
                            err_msg=err_msg(actual=epoch_data,
                                            desired=metric.epoch_data,
                                            name=metric.name + ": Epoch Data"))


