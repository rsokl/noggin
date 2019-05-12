from itertools import combinations

from hypothesis import given

import tests.custom_strategies as cst
from liveplot.logger import LiveLogger, LiveMetric
from liveplot.plotter import LivePlot
from liveplot.utils import check_valid_color


@given(cst.choices("abcdefg", 3))
def test_choices(choice):
    assert choice in set(combinations("abcdefg", 3))


@given(logger=cst.loggers())
def test_loggers(logger: LiveLogger):
    """Ensure that loggers() can produce a Logger that can round-trip"""
    LiveLogger.from_dict(logger.to_dict())


@given(plotter=cst.plotters())
def test_plotters(plotter: LivePlot):
    """Ensure that loggers() can produce a Logger that can round-trip"""
    LivePlot.from_dict(plotter.to_dict())


@given(metrics_dict=cst.metric_dict("metric-a"))
def test_metrics_dict(metrics_dict: dict):
    """Ensure that metrics_dict() can round-trip via LiveMetric"""
    LiveMetric.from_dict(metrics_dict).to_dict()


@given(live_metrics=cst.live_metrics())
def test_livemetrics(live_metrics: dict):
    """Ensure that each entry in live_metrics() can round-trip via LiveMetric"""
    for metric_name, metrics_dict in live_metrics.items():
        LiveMetric.from_dict(metrics_dict).to_dict()


@given(cst.matplotlib_colors())
def test_colors(color):
    check_valid_color(color)
