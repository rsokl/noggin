import pytest
from math import isclose
from liveplot import LivePlot
from time import sleep, time


class ControlledPlot(LivePlot):
    """Mocks out plotting. Forces plotting to take a specified
    time; no plots are created"""

    def __init__(self, *args, plot_time, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_time = plot_time
        self._liveplot = True

    def _init_plot_window(self):
        return

    def plot(self):
        sleep(self.plot_time)


@pytest.mark.parametrize("plot_time", [0.0001, 0.001, 0.01])
@pytest.mark.parametrize("outer_time", [0.0001, 0.001])
@pytest.mark.parametrize("max_fraction", [0.0, 0.01, 0.3, 0.5])
def test_adaptive_plotting(plot_time, outer_time, max_fraction):
    plotter = ControlledPlot(metrics="a", plot_time=plot_time)

    plotter._max_fraction_spent_plotting = max_fraction
    total_plot_time = 0.0

    start_time = time()
    for n in range(500):
        # continue
        start_plot = time()
        plotter.set_train_batch(dict(a=1), batch_size=1, plot=True)
        total_plot_time += time() - start_plot
        sleep(outer_time)
    total_time = time() - start_time

    actual_fraction = total_plot_time / total_time

    assert isclose(actual_fraction, max_fraction, rel_tol=0.2, abs_tol=0.01)


@pytest.mark.parametrize(
    ("plot_time", "outer_time", "max_fraction", "expected_fraction"),
    [(0.1, 0.001, 1.0, 1.0), (0.1, 0.001, 0.7, 0.7), (0.001, 0.001, 1.0, 0.5)],
)
def test_exhaustive_plotting(plot_time, outer_time, max_fraction, expected_fraction):
    """Ensure that plotting dominates runtime when `max_fraction_spent_plotting` is 1
    and plotting is dominant.

    If plotting and outer-loop time are comparable, the fraction of time spent
    plotting should be ~0.5"""

    plotter = ControlledPlot(metrics="a", plot_time=plot_time)
    plotter._max_fraction_spent_plotting = max_fraction
    total_plot_time = 0.0

    start_time = time()
    for n in range(50):
        # continue
        start_plot = time()
        plotter.set_train_batch(dict(a=1), batch_size=1, plot=True)
        total_plot_time += time() - start_plot
        sleep(outer_time)
    total_time = time() - start_time

    actual_fraction = total_plot_time / total_time

    assert isclose(actual_fraction, expected_fraction, rel_tol=0.1, abs_tol=0.01)