from tests import err_msg

from liveplot.logger import LiveLogger, LiveMetric

from typing import List

from numpy.testing import assert_array_equal, assert_allclose

from hypothesis import settings
import hypothesis.strategies as st
from hypothesis.strategies import SearchStrategy
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule, precondition


@settings(max_examples=500)
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
        train_metric_names = ["a", "b", "c"][:num_train_metrics]
        for name in train_metric_names:
            self.train_metrics.append(LiveMetric(name=name))

        test_metric_names = ["a", "b", "c"][:num_test_metrics]
        for name in test_metric_names:
            self.test_metrics.append(LiveMetric(name=name))

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


TestLiveLogger = LiveLoggerStateMachine.TestCase
