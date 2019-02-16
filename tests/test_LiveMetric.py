from liveplot.logger import LiveMetric

from typing import Optional

import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from hypothesis import given, settings
import hypothesis.strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule


# @given(name=st.sampled_from([1, None, np.array([1]), ["moo"]]))
# def test_badname(name: str):
#     with pytest.raises(TypeError):
#         LiveMetric(name)


@settings(max_examples=1000)
class LiveMetricChecker(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()

        self.batch_data = []
        self._weights = []
        self.epoch_data = []
        self.epoch_domain = []

    @initialize(name=st.sampled_from(["a", "b", "c"]))
    def init_metric(self, name: str):
        self.livemetric = LiveMetric(name)
        self.name = name

    @rule(value=st.floats(-1e6, 1e6),
          weighting=st.one_of(st.none(), st.floats(0, 2)))
    def add_datapoint(self, value: float, weighting: Optional[float]):
        if weighting is not None:
            self.livemetric.add_datapoint(value=value, weighting=weighting)
        else:
            self.livemetric.add_datapoint(value=value)
        self.batch_data.append(value)
        self._weights.append(weighting if weighting is not None else 1.)

    @rule()
    def set_epoch_datapoint(self):
        self.livemetric.set_epoch_datapoint()

        if self._weights:
            batch_dat = np.array(self.batch_data)[-len(self._weights):]
            weights = np.array(self._weights) / sum(self._weights)
            weights = np.nan_to_num(weights)
            epoch_mean = batch_dat @ weights
            self.epoch_data.append(epoch_mean)
            self.epoch_domain.append(len(self.batch_data))
            self._weights = []

    @rule()
    def show_repr(self):
        """ Ensure no side effects of calling `repr()`"""
        repr(self.livemetric)

    @rule()
    def to_dict(self):
        """ Ensure no side effects of calling `to_dict()`"""
        self.livemetric.to_dict()

    @rule()
    def compare(self):
        batch_domain = np.arange(1, len(self.batch_data) + 1)
        batch_data = np.asarray(self.batch_data)
        epoch_data = np.asarray(self.epoch_data)
        epoch_domain = np.asarray(self.epoch_domain)

        assert_array_equal(batch_data, self.livemetric.batch_data)
        assert_array_equal(epoch_domain, self.livemetric.epoch_domain)
        assert_array_equal(self.livemetric.batch_domain, batch_domain)
        assert_allclose(actual=self.livemetric.epoch_data,
                        desired=epoch_data)

        assert self.livemetric.name == self.name

# class Moo(RuleBasedStateMachine):
#     def __init__(self):
#         super().__init__()
#         self.weighted_sum = 0
#         self.weights = []
#
#     @rule(value=st.floats(0, 2))
#     def add_data(self, value): pass


#TestMoo = Moo.TestCase

TestLiveMetricChecker = LiveMetricChecker.TestCase