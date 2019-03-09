from tests.utils import err_msg

from liveplot.logger import LiveMetric

from typing import Optional, Any

import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from hypothesis import given
import hypothesis.strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule


@given(name=st.sampled_from([1, None, np.array([1]), ["moo"]]))
def test_badname(name: Any):
    with pytest.raises(TypeError):
        LiveMetric(name)


def test_trivial_case():
    """ Perform a trivial sanity check on live metric"""
    metric = LiveMetric("a")
    metric.add_datapoint(1., weighting=1.)
    metric.add_datapoint(3., weighting=1.)
    metric.set_epoch_datapoint(99)
    assert_array_equal(metric.batch_domain, np.array([1, 2]))
    assert_array_equal(metric.batch_data, np.array([1., 3.]))

    assert_array_equal(metric.epoch_domain, np.array([99]))
    assert_array_equal(metric.epoch_data, np.array([1. / 2. + 3. / 2.]))

    dict_ = metric.to_dict()
    for name in ("batch_data", "epoch_data", "epoch_domain"):
        assert_array_equal(dict_[name], getattr(metric, name),
                           err_msg=name + " does not map to the "
                                          "correct value in the metric-dict")


class LiveMetricChecker(RuleBasedStateMachine):
    """ Ensures that exercising the api of LiveMetric produces
    results that are consistent with a simplistic implementation"""
    def __init__(self):
        super().__init__()

        self.batch_data = []
        self._weights = []
        self.epoch_data = []
        self.epoch_domain = []
        self.livemetric = None  # type: LiveMetric
        self.name = None  # type: str

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

        assert_array_equal(batch_data,
                           self.livemetric.batch_data,
                           err_msg=err_msg(actual=batch_data,
                                           desired=self.livemetric.batch_data,
                                           name="Batch Data"))
        assert_array_equal(epoch_domain,
                           self.livemetric.epoch_domain,
                           err_msg=err_msg(actual=epoch_domain,
                                           desired=self.livemetric.epoch_domain,
                                           name="Epoch Domain"))
        assert_array_equal(self.livemetric.batch_domain,
                           batch_domain,
                           err_msg=err_msg(actual=batch_domain,
                                           desired=self.livemetric.batch_domain,
                                           name="Batch Domain"))
        assert_allclose(actual=self.livemetric.epoch_data,
                        desired=epoch_data,
                        err_msg=err_msg(actual=epoch_data,
                                        desired=self.livemetric.epoch_data,
                                        name="Epoch Data"))

        assert self.livemetric.name == self.name


TestLiveMetricChecker = LiveMetricChecker.TestCase
