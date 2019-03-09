from typing import Dict, Sequence
from numpy import ndarray

from liveplot.logger import LiveMetric

from numpy.testing import assert_array_equal

Metrics = Dict[str, Dict[str, ndarray]]


def err_msg(actual, desired, name):
    return (f"{name} does not match."
            f"\nExpected:"
            f"\n\t{repr(desired)}"
            f"\nGot:"
            f"\n\t{repr(actual)}")


def compare_all_metrics(x: Metrics, y: Metrics):
    assert sorted(x) == sorted(y), "The metric names do not match"

    for metric_name in x:
        x_batch_data = x[metric_name]["batch_data"]
        x_epoch_domain = x[metric_name]["epoch_domain"]
        x_epoch_data = x[metric_name]["epoch_data"]

        y_batch_data = y[metric_name]["batch_data"]
        y_epoch_domain = y[metric_name]["epoch_domain"]
        y_epoch_data = y[metric_name]["epoch_data"]

        assert_array_equal(x_batch_data, y_batch_data,
                           err_msg=metric_name + ": batch data")
        assert_array_equal(x_epoch_domain, y_epoch_domain,
                           err_msg=metric_name + ": epoch domain")

        assert_array_equal(x_epoch_data, y_epoch_data,
                           err_msg=metric_name + ": epoch data")
