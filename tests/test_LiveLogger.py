from tests.utils import compare_all_metrics

from liveplot import save_metrics, load_metrics
from liveplot.logger import LiveLogger, LiveMetric

from typing import List

from numpy.testing import assert_array_equal

import hypothesis.strategies as st
from hypothesis.strategies import SearchStrategy
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule, precondition
from hypothesis import note

import numpy as np

import pytest


def test_trivial_case():
    """ Perform a trivial sanity check on live logger"""
    logger = LiveLogger()
    logger.set_train_batch(dict(a=1.), batch_size=1)
    logger.set_train_batch(dict(a=3.), batch_size=1)
    logger.set_train_epoch()

    assert_array_equal(logger.train_metrics['a']['batch_data'], np.array([1., 3.]))
    assert_array_equal(logger.train_metrics['a']['epoch_domain'], np.array([2]))
    assert_array_equal(logger.train_metrics['a']['epoch_data'], np.array([1. / 2. + 3. / 2.]))


class LiveLoggerStateMachine(RuleBasedStateMachine):
    """ Ensures that exercising the api of LiveLogger produces
    results that are consistent with a simplistic implementation"""
    def __init__(self):
        super().__init__()
        self.train_metrics = []  # type: List[LiveMetric]
        self.test_metrics = []  # type: List[LiveMetric]
        self.logger = LiveLogger()
        self.train_batch_set = False
        self.test_batch_set = False
        self.num_train_batch = 0

    @initialize(num_train_metrics=st.integers(0, 3),
                num_test_metrics=st.integers(0, 3))
    def choose_metrics(self,
                       num_train_metrics: int,
                       num_test_metrics: int):
        train_metric_names = ["metric-a", "metric-b", "metric-c"][:num_train_metrics]
        for name in train_metric_names:
            self.train_metrics.append(LiveMetric(name=name))

        test_metric_names = ["metric-a", "metric-b", "metric-c"][:num_test_metrics]
        for name in test_metric_names:
            self.test_metrics.append(LiveMetric(name=name))

        note("Train metric names: {}".format(train_metric_names))
        note("Test metric names: {}".format(test_metric_names))

    @rule()
    def get_repr(self):
        """ Ensure no side effect """
        repr(self.logger)

    @rule(batch_size=st.integers(0, 2), data=st.data())
    def set_train_batch(self, batch_size: int, data: SearchStrategy):
        if self.train_metrics:
            self.num_train_batch += 1

        self.train_batch_set = True
        batch = {metric.name: data.draw(st.floats(-1, 1), label=metric.name)
                 for metric in self.train_metrics}
        self.logger.set_train_batch(metrics=batch, batch_size=batch_size)

        for metric in self.train_metrics:
            metric.add_datapoint(batch[metric.name], weighting=batch_size)

    @rule()
    def set_train_epoch(self):
        self.logger.set_train_epoch()
        for metric in self.train_metrics:
            metric.set_epoch_datapoint()
            
    @rule(batch_size=st.integers(0, 2), data=st.data())
    def set_test_batch(self, batch_size: int, data: SearchStrategy):
        self.test_batch_set = True
        batch = {metric.name: data.draw(st.floats(-1, 1), label=metric.name)
                 for metric in self.test_metrics}
        self.logger.set_test_batch(metrics=batch, batch_size=batch_size)

        for metric in self.test_metrics:
            metric.add_datapoint(batch[metric.name], weighting=batch_size)

    @rule()
    def set_test_epoch(self):
        self.logger.set_test_epoch()

        # align test-epoch with train domain
        for metric in self.test_metrics:
            if metric.name in {m.name for m in self.train_metrics}:
                x = self.num_train_batch if self.num_train_batch > 0 else None
            else:
                x = None
            metric.set_epoch_datapoint(x)
            
    @precondition(lambda self: self.train_batch_set)
    @rule()
    def compare_train_metrics(self):
        logged_metrics = self.logger.train_metrics
        expected_metrics = dict((metric.name, metric.to_dict()) for metric in self.train_metrics)
        compare_all_metrics(logged_metrics, expected_metrics)

    @precondition(lambda self: self.test_batch_set)
    @rule()
    def compare_test_metrics(self):
        logged_metrics = self.logger.test_metrics
        expected_metrics = dict((metric.name, metric.to_dict()) for metric in self.test_metrics)
        compare_all_metrics(logged_metrics, expected_metrics)

    @rule(save_via_plot_object=st.booleans())
    def check_metric_io(self, save_via_plot_object: bool):
        """Ensure the saving/loading metrics always produces self-consistent
        results with the logger"""
        from uuid import uuid4
        filename = str(uuid4())
        if save_via_plot_object:
            save_metrics(filename, liveplot=self.logger)
        else:
            save_metrics(filename,
                         train_metrics=self.logger.train_metrics,
                         test_metrics=self.logger.test_metrics)
        io_train_metrics, io_test_metrics = load_metrics(filename)

        compare_all_metrics(io_train_metrics, self.logger.train_metrics)
        compare_all_metrics(io_test_metrics, self.logger.test_metrics)


@pytest.mark.usefixtures("cleandir")
class TestLiveLogger(LiveLoggerStateMachine.TestCase):
    pass
