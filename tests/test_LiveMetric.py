from numbers import Real
from random import Random
from typing import Any, Optional

import hypothesis.strategies as st
import numpy as np
import pytest
import tests.custom_strategies as cst
from hypothesis import given, settings, HealthCheck
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)
from liveplot.logger import LiveMetric
from numpy.testing import assert_allclose, assert_array_equal
from tests.utils import err_msg


@given(name=st.sampled_from([1, None, np.array([1]), ["moo"]]))
def test_badname(name: Any):
    with pytest.raises(TypeError):
        LiveMetric(name)


def test_trivial_case():
    """ Perform a trivial sanity check on live metric"""
    metric = LiveMetric("a")
    metric.add_datapoint(1.0, weighting=1.0)
    metric.add_datapoint(3.0, weighting=1.0)
    metric.set_epoch_datapoint(99)
    assert_array_equal(metric.batch_domain, np.array([1, 2]))
    assert_array_equal(metric.batch_data, np.array([1.0, 3.0]))

    assert_array_equal(metric.epoch_domain, np.array([99]))
    assert_array_equal(metric.epoch_data, np.array([1.0 / 2.0 + 3.0 / 2.0]))

    dict_ = metric.to_dict()
    for name in ("batch_data", "epoch_data", "epoch_domain"):
        assert_array_equal(
            dict_[name],
            getattr(metric, name),
            err_msg=name + " does not map to the correct value in the metric-dict",
        )


@pytest.mark.parametrize(
    "bad_input", [cst.everything_except(dict), st.just(dict(a_bad_key=1))]
)
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_from_dict_input_validation(bad_input: st.SearchStrategy, data: st.DataObject):
    bad_input = data.draw(bad_input, label="bad_input")
    with pytest.raises((ValueError, TypeError)):
        LiveMetric.from_dict(bad_input)


static_logger_dict = cst.live_metrics(min_num_metrics=1).example(
    random=Random(0)
)  # type: dict


@pytest.mark.parametrize(
    "bad_input",
    [
        dict(batch_data=np.arange(9).reshape(3, 3)),
        dict(epoch_data=np.arange(9).reshape(3, 3)),
        dict(cnt_since_epoch=-1),
        dict(epoch_domain=np.arange(9).reshape(3, 3)),
        dict(batch_data=cst.everything_except(np.ndarray)),
        dict(epoch_data=cst.everything_except(np.ndarray)),
        dict(epoch_domain=cst.everything_except(np.ndarray)),
        dict(cnt_since_epoch=cst.everything_except(Real)),
        dict(total_weighting=cst.everything_except(Real)),
        dict(running_weighted_sum=cst.everything_except(Real)),
        dict(name=cst.everything_except(str)),
    ],
)
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_from_dict_input_validation2(bad_input: dict, data: st.DataObject):
    input_dict = {}

    bad_input = {
        k: data.draw(v, label=k) if isinstance(v, st.SearchStrategy) else v
        for k, v in bad_input.items()
    }
    for name, metrics in static_logger_dict.items():
        input_dict = metrics.copy()
        input_dict.update(bad_input)
        break

    assert input_dict

    with pytest.raises((ValueError, TypeError)):
        LiveMetric.from_dict(input_dict)


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

    @rule(value=st.floats(-1e6, 1e6), weighting=st.one_of(st.none(), st.floats(0, 2)))
    def add_datapoint(self, value: float, weighting: Optional[float]):
        if weighting is not None:
            self.livemetric.add_datapoint(value=value, weighting=weighting)
        else:
            self.livemetric.add_datapoint(value=value)
        self.batch_data.append(value)
        self._weights.append(weighting if weighting is not None else 1.0)

    @rule()
    def set_epoch_datapoint(self):
        self.livemetric.set_epoch_datapoint()

        if self._weights:
            batch_dat = np.array(self.batch_data)[-len(self._weights) :]
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

    @precondition(lambda self: self.livemetric is not None)
    @invariant()
    def dict_roundtrip(self):
        """Ensure `from_dict(to_dict())` round trip is successful"""
        metrics_dict = self.livemetric.to_dict()
        new_metrics = LiveMetric.from_dict(metrics_dict=metrics_dict)

        for attr in [
            "name",
            "batch_data",
            "epoch_data",
            "epoch_domain",
            "_batch_data",
            "_epoch_data",
            "_epoch_domain",
            "_running_weighted_sum",
            "_total_weighting",
            "_cnt_since_epoch",
        ]:
            desired = getattr(self.livemetric, attr)
            actual = getattr(new_metrics, attr)
            assert type(actual) == type(desired), attr
            assert_array_equal(
                actual,
                desired,
                err_msg="`LiveMetric.from_dict` did not round-trip successfully.\n"
                "livemetric.{} does not match.\nGot: {}\nExpected: {}"
                "".format(attr, actual, desired),
            )

    @rule()
    def check_batch_data_is_consistent(self):
        actual_batch_data1 = self.livemetric.batch_data
        actual_batch_data2 = self.livemetric.batch_data
        assert isinstance(actual_batch_data1, np.ndarray)
        assert isinstance(actual_batch_data2, np.ndarray)
        assert_array_equal(
            actual_batch_data1,
            actual_batch_data2,
            err_msg="calling `LiveMetric.batch_data` two"
            "consecutive times produces different "
            "results",
        )

    @rule()
    def check_epoch_data_is_consistent(self):
        actual_epoch_data1 = self.livemetric.epoch_data
        actual_epoch_data2 = self.livemetric.epoch_data
        assert isinstance(actual_epoch_data1, np.ndarray)
        assert isinstance(actual_epoch_data2, np.ndarray)
        assert_array_equal(
            actual_epoch_data1,
            actual_epoch_data2,
            err_msg="calling `LiveMetric.epoch_data` two"
            "consecutive times produces different "
            "results",
        )

    @rule()
    def check_epoch_domain_is_consistent(self):
        actual_epoch_domain1 = self.livemetric.epoch_domain
        actual_epoch_domain2 = self.livemetric.epoch_domain
        assert isinstance(actual_epoch_domain1, np.ndarray)
        assert isinstance(actual_epoch_domain2, np.ndarray)
        assert_array_equal(
            actual_epoch_domain1,
            actual_epoch_domain2,
            err_msg="calling `LiveMetric.epoch_domain` two"
            "consecutive times produces different "
            "results",
        )

    @precondition(lambda self: self.livemetric is not None)
    @invariant()
    def compare(self):
        expected_batch_domain = np.arange(1, len(self.batch_data) + 1)
        expected_batch_data = np.asarray(self.batch_data)
        expected_epoch_data = np.asarray(self.epoch_data)
        expected_epoch_domain = np.asarray(self.epoch_domain)

        actual_batch_domain = self.livemetric.batch_domain
        actual_batch_data = self.livemetric.batch_data
        actual_epoch_data = self.livemetric.epoch_data
        actual_epoch_domain = self.livemetric.epoch_domain

        assert isinstance(actual_batch_domain, np.ndarray)
        assert isinstance(actual_batch_data, np.ndarray)
        assert isinstance(actual_epoch_data, np.ndarray)
        assert isinstance(actual_epoch_domain, np.ndarray)

        assert_array_equal(
            expected_batch_data,
            actual_batch_data,
            err_msg=err_msg(
                desired=expected_batch_data, actual=actual_batch_data, name="Batch Data"
            ),
        )
        assert_array_equal(
            expected_epoch_domain,
            self.livemetric.epoch_domain,
            err_msg=err_msg(
                desired=expected_epoch_domain,
                actual=self.livemetric.epoch_domain,
                name="Epoch Domain",
            ),
        )
        assert_array_equal(
            self.livemetric.batch_domain,
            actual_batch_domain,
            err_msg=err_msg(
                desired=expected_batch_domain,
                actual=actual_batch_domain,
                name="Batch Domain",
            ),
        )
        assert_allclose(
            actual=actual_epoch_data,
            desired=expected_epoch_data,
            err_msg=err_msg(
                desired=expected_epoch_data, actual=actual_batch_data, name="Epoch Data"
            ),
        )

        assert self.livemetric.name == self.name


TestLiveMetricChecker = LiveMetricChecker.TestCase
