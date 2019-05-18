from collections import OrderedDict
from contextlib import contextmanager

import hypothesis.strategies as st
from hypothesis import assume, note, settings
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule
from hypothesis.strategies import SearchStrategy
from matplotlib.pyplot import close

import tests.custom_strategies as cst
from liveplot.logger import LiveLogger
from liveplot.plotter import LivePlot


@contextmanager
def close_fig(fig):
    try:
        yield None
    finally:
        close(fig)


@settings(deadline=None)
class LivePlotStateMachine(RuleBasedStateMachine):
    """Provides basic rules for exercising essential aspects of LivePlot"""

    def __init__(self):
        super().__init__()
        self.train_metric_names = []
        self.test_metric_names = []
        self.train_batch_set = False
        self.test_batch_set = False
        self.plotter = None  # type: LivePlot
        self.logger = None  # type: LiveLogger

    @initialize(
        num_train_metrics=st.integers(0, 3),
        num_test_metrics=st.integers(0, 3),
        data=st.data(),
    )
    def choose_metrics(
        self, num_train_metrics: int, num_test_metrics: int, data: st.SearchStrategy
    ):
        assume(num_train_metrics + num_test_metrics > 0)
        self.train_metric_names = ["metric-a", "metric-b", "metric-c"][
            :num_train_metrics
        ]

        self.test_metric_names = ["metric-a", "metric-b", "metric-c"][:num_test_metrics]
        train_colors = data.draw(
            st.lists(
                cst.matplotlib_colors(),
                min_size=num_train_metrics,
                max_size=num_train_metrics,
            ),
            label="train_colors",
        )

        test_colors = data.draw(
            st.lists(
                cst.matplotlib_colors(),
                min_size=num_test_metrics,
                max_size=num_test_metrics,
            ),
            label="test_colors",
        )

        metrics = OrderedDict(
            (n, dict())
            for n in sorted(set(self.train_metric_names + self.test_metric_names))
        )

        for metric, color in zip(self.train_metric_names, train_colors):
            metrics[metric]["train"] = color

        for metric, color in zip(self.test_metric_names, test_colors):
            metrics[metric]["test"] = color

        self.plotter = LivePlot(metrics, refresh=-1)
        self.logger = LiveLogger()

        note("Train metric names: {}".format(self.train_metric_names))
        note("Test metric names: {}".format(self.test_metric_names))

    @rule(batch_size=st.integers(0, 2), data=st.data(), plot=st.booleans())
    def set_train_batch(self, batch_size: int, data: SearchStrategy, plot: bool):
        self.train_batch_set = True

        batch = {
            name: data.draw(st.floats(-1, 1), label=name)
            for name in self.train_metric_names
        }
        self.logger.set_train_batch(metrics=batch, batch_size=batch_size)
        self.plotter.set_train_batch(metrics=batch, batch_size=batch_size, plot=plot)

    @rule()
    def set_train_epoch(self):
        self.logger.set_train_epoch()
        self.plotter.plot_train_epoch()

    @rule(batch_size=st.integers(0, 2), data=st.data())
    def set_test_batch(self, batch_size: int, data: SearchStrategy):
        self.test_batch_set = True

        batch = {
            name: data.draw(st.floats(-1, 1), label=name)
            for name in self.test_metric_names
        }
        self.logger.set_test_batch(metrics=batch, batch_size=batch_size)
        self.plotter.set_test_batch(metrics=batch, batch_size=batch_size)

    @rule()
    def set_test_epoch(self):
        self.logger.set_test_epoch()
        self.plotter.plot_test_epoch()

    def teardown(self):
        if self.plotter is not None:
            fig, _ = self.plotter.plot_objects
            if fig is not None:
                close(fig)
        super().teardown()
