from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import pytest

from liveplot import create_plot, LivePlot
from liveplot.typing import Figure, ndarray, Axes
from liveplot.plotter import _check_valid_color
from tests.utils import compare_all_metrics, err_msg
from tests import close_plots


@pytest.mark.parametrize(
    "color",
    (
        "r",
        "red",
        "C0",
        "#eeefff",
        "burlywood",
        "0.25",
        (0.1, 0.2, 0.3),
        (0.1, 0.2, 0.3, 0.2),
    ),
)
def test_valid_colors(color):
    assert _check_valid_color(color)


@pytest.mark.parametrize(
    "color", ("helloworld", "", "#fff", (0.1, 0.1), (0.1, 0.2, 0.3, 1.0, 2.0))
)
def test_invalid_colors(color):
    with pytest.raises(ValueError):
        _check_valid_color(color)


def test_err_msg():
    name = "moo"
    actual = 1
    desired = 2
    msg = err_msg(actual, desired, name)
    assert name in msg
    assert msg.split().index("Expected:") + 1 == msg.split().index(
        repr(desired)
    ), "'Expected' should be followed by the expected value"
    assert msg.split().index("Got:") + 1 == msg.split().index(
        repr(actual)
    ), "'Got' should be followed by the actual value"


def test_mismatched_number_of_metrics():
    with pytest.raises(AssertionError):
        compare_all_metrics(
            dict(
                a=dict(
                    batch_data=np.array([0]),
                    epoch_data=np.array([0]),
                    epoch_domain=np.array([0]),
                )
            ),
            dict(),
        )


@pytest.mark.parametrize(
    "mismatched_category", (None, "batch_data", "epoch_data", "epoch_domain")
)
@pytest.mark.parametrize("mismatched_metric", ("a", "b"))
def test_mismatched_metrics(mismatched_category, mismatched_metric):
    x = dict(
        a=dict(
            batch_data=np.array([0]),
            epoch_data=np.array([0]),
            epoch_domain=np.array([0]),
        ),
        b=dict(
            batch_data=np.array([0]),
            epoch_data=np.array([0]),
            epoch_domain=np.array([0]),
        ),
    )

    y = dict(
        a=dict(
            batch_data=np.array([0]),
            epoch_data=np.array([0]),
            epoch_domain=np.array([0]),
        ),
        b=dict(
            batch_data=np.array([0]),
            epoch_data=np.array([0]),
            epoch_domain=np.array([0]),
        ),
    )
    if mismatched_category is not None:
        x[mismatched_metric][mismatched_category] += 1
        with pytest.raises(AssertionError):
            compare_all_metrics(x, y)
    else:
        compare_all_metrics(x, y)


@given(
    metrics=st.lists(st.sampled_from("abcdef"), min_size=1, unique=True).map(tuple),
    figsize=st.tuples(*[st.floats(min_value=1, max_value=10)] * 2),
    refresh=st.floats(0.5, 2),
)
def test_create_plot(metrics, figsize, refresh):
    with close_plots():
        plotter, fig, ax = create_plot(
            metrics=metrics, figsize=figsize, refresh=refresh
        )
        assert isinstance(plotter, LivePlot)
        assert isinstance(fig, Figure)
        assert isinstance(ax, (Axes, ndarray))
