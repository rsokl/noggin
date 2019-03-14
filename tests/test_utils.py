from tests.utils import err_msg, compare_all_metrics
from liveplot.utils import check_valid_color

import numpy as np

import pytest


@pytest.mark.parametrize(
    "color", ("r",
              "red",
              "C0",
              "#eeefff",
              "burlywood",
              "0.25",
              (.1, .2, .3),
              (.1, .2, .3, .2),
              )
)
def test_valid_colors(color):
    assert check_valid_color(color)


@pytest.mark.parametrize(
    "color", ("helloworld",
              "",
              "#fff",
              None,
              (.1, .1),
              (.1, .2, .3, 1., 2.),
              )
)
def test_invalid_colors(color):
    with pytest.raises(ValueError):
        check_valid_color(color)


def test_err_msg():
    name = "moo"
    actual = 1
    desired = 2
    msg = err_msg(actual, desired, name)
    assert name in msg
    assert msg.split().index("Expected:") + 1 == msg.split().index(repr(desired)), \
        "'Expected' should be followed by the expected value"
    assert msg.split().index("Got:") + 1 == msg.split().index(repr(actual)), \
        "'Got' should be followed by the actual value"


def test_mismatched_number_of_metrics():
    with pytest.raises(AssertionError):
        compare_all_metrics(dict(a=dict(batch_data=np.array([0]),
                                        epoch_data=np.array([0]),
                                        epoch_domain=np.array([0]))),
                            dict())


@pytest.mark.parametrize("mismatched_category",
                         (None, "batch_data", "epoch_data", "epoch_domain"))
@pytest.mark.parametrize("mismatched_metric",
                         ("a", "b"))
def test_mismatched_metrics(mismatched_category, mismatched_metric):
    x = dict(a=dict(batch_data=np.array([0]),
                    epoch_data=np.array([0]),
                    epoch_domain=np.array([0])),
             b=dict(batch_data=np.array([0]),
                    epoch_data=np.array([0]),
                    epoch_domain=np.array([0]))
             )

    y = dict(a=dict(batch_data=np.array([0]),
                    epoch_data=np.array([0]),
                    epoch_domain=np.array([0])),
             b=dict(batch_data=np.array([0]),
                    epoch_data=np.array([0]),
                    epoch_domain=np.array([0]))
             )
    if mismatched_category is not None:
        x[mismatched_metric][mismatched_category] += 1
        with pytest.raises(AssertionError):
            compare_all_metrics(x, y)
    else:
        compare_all_metrics(x, y)
