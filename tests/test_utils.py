import string
from typing import Optional

import hypothesis.strategies as st
import numpy as np
import pytest
import tests.custom_strategies as cst
from hypothesis import example, given, settings
from numpy.testing import assert_array_equal
from tests import close_plots
from tests.utils import compare_all_metrics, err_msg
from xarray.testing import assert_equal

from noggin import LiveLogger, LivePlot, create_plot
from noggin.plotter import _check_valid_color
from noggin.typing import Axes, Figure, ndarray
from noggin.utils import load_metrics, plot_logger, save_metrics


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


@settings(deadline=None)
@given(
    metrics=st.lists(st.sampled_from("abcdef"), min_size=1, unique=True).map(tuple),
    figsize=st.tuples(*[st.floats(min_value=1, max_value=10)] * 2),
    max_fraction_spent_plotting=st.floats(0.0, 1.0),
    last_n_batches=st.none() | st.integers(1, 10),
)
def test_create_plot(metrics, figsize, max_fraction_spent_plotting, last_n_batches):
    with close_plots():
        plotter, fig, ax = create_plot(
            metrics=metrics,
            figsize=figsize,
            max_fraction_spent_plotting=max_fraction_spent_plotting,
            last_n_batches=last_n_batches,
        )
        assert isinstance(plotter, LivePlot)
        assert isinstance(fig, Figure)
        assert isinstance(ax, (Axes, ndarray))
        assert plotter.last_n_batches == last_n_batches
        assert plotter.max_fraction_spent_plotting == max_fraction_spent_plotting


@example(metric_name="a;a")  # tests separator collision
@given(metric_name=st.text(alphabet=string.printable, min_size=1))
@pytest.mark.usefixtures("cleandir")
def test_metric_io_train(metric_name: str):
    logger = LiveLogger()
    logger.set_train_batch({metric_name: 1}, batch_size=1)
    save_metrics("test.pkl", train_metrics=logger.train_metrics)
    train, test = load_metrics("test.pkl")

    assert list(train) == list(logger.train_metrics)
    for k, actual in train[metric_name].items():
        expected = logger.train_metrics[metric_name][k]
        if isinstance(actual, np.ndarray):
            assert_array_equal(actual, expected)
        else:
            assert expected == actual


@example(metric_name="a;a")
@given(metric_name=st.text(alphabet=string.printable, min_size=1))
@pytest.mark.usefixtures("cleandir")
def test_metric_io_test(metric_name: str):
    logger = LiveLogger()
    logger.set_test_batch({metric_name: 1}, batch_size=1)
    save_metrics("test.pkl", test_metrics=logger.test_metrics)
    train, test = load_metrics("test.pkl")

    assert list(test) == list(logger.test_metrics)
    for k, actual in test[metric_name].items():
        expected = logger.test_metrics[metric_name][k]
        if isinstance(actual, np.ndarray):
            assert_array_equal(actual, expected)
        else:
            assert expected == actual


@settings(deadline=None)
@given(bad_logger=cst.everything_except(LiveLogger))
def test_plot_logger_validation(bad_logger):
    with pytest.raises(TypeError):
        plot_logger(bad_logger)


@settings(deadline=None)
@given(
    logger=cst.loggers(min_num_metrics=1),
    last_n_batches=st.none() | st.integers(1, 10),
    plot_batches=st.booleans(),
    data=st.data(),
)
def test_plot_logger(
    logger: LiveLogger,
    last_n_batches: Optional[int],
    plot_batches: bool,
    data: st.DataObject,
):
    metrics = set(list(logger.train_metrics.keys()) + list(logger.test_metrics.keys()))

    colors = data.draw(
        st.none()
        | st.fixed_dictionaries(
            {
                k: st.fixed_dictionaries(
                    {"train": cst.matplotlib_colors(), "test": cst.matplotlib_colors()}
                )
                for k in metrics
            }
        ),
        label="colors",
    )
    with close_plots():
        plotter, fig, ax = plot_logger(
            logger,
            colors=colors,
            plot_batches=plot_batches,
            last_n_batches=last_n_batches,
        )
        assert isinstance(plotter, LivePlot)
        assert isinstance(fig, Figure)
        assert isinstance(ax, (Axes, ndarray))
    assert plotter.last_n_batches == last_n_batches
    assert plotter.metric_colors == (colors if colors else {})

    train_batch_expected, train_epoch_expected = logger.to_xarray("train")
    test_batch_expected, test_epoch_expected = logger.to_xarray("test")

    train_batch_actual, train_epoch_actual = plotter.to_xarray("train")
    test_batch_actual, test_epoch_actual = plotter.to_xarray("test")

    assert_equal(train_batch_actual, train_batch_expected)
    assert_equal(test_batch_actual, test_batch_expected)
    assert_equal(train_epoch_actual, train_epoch_expected)
    assert_equal(test_epoch_actual, test_epoch_expected)
