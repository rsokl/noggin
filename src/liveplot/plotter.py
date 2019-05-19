import importlib
import time
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from inspect import cleandoc
from itertools import product
from numbers import Integral, Real
from typing import Dict, Optional, Tuple, Union
from warnings import warn

import numpy as np
from liveplot.logger import LiveLogger, LiveMetric
from liveplot.typing import Metrics, ValidColor
from matplotlib.pyplot import Axes, Figure

__all__ = ["LivePlot"]


def _check_valid_color(c: ValidColor) -> bool:
    """
    Checks if `c` is a valid color argument for matplotlib or `None`.
    Raises `ValueError` if `c` is not a valid color.

    Parameters
    ----------
    c : Union[str, Real, Sequence[Real], NoneType]

    Returns
    -------
    bool

    Raises
    ------
    ValueError"""
    from matplotlib.colors import is_color_like

    if c is not None and not is_color_like(c):
        raise ValueError("{} is not a valid matplotlib color".format(repr(c)))
    else:
        return True


def _filter_data(seq, cap=2000):
    step = max(len(seq) // cap, 1)
    return seq[::step]


class LivePlot(LiveLogger):
    """ Plots batch-level and epoch-level summary statistics of the training and
        testing metrics of a model during a session.

        Notes
        -----
        Live plotting is only supported for the 'nbAgg' backend (i.e.
        when the cell magic ``%matplotlib notebook`` is invoked in a
        jupyter notebook). """

    @property
    def metric_colors(self) -> Dict[str, Dict[str, str]]:
        """ Returns
            -------
            Dict[str, Dict[str, color-value]]
                {'<metric-name>' -> {'train'/'test' -> color-value}}"""
        out = defaultdict(dict)
        for k, v in self._train_colors.items():
            out[k]["train"] = v

        for k, v in self._test_colors.items():
            out[k]["test"] = v
        return dict(out)

    @property
    def refresh(self) -> float:
        """ The minimum time between canvas-draw events, in seconds.
            A negative `refresh` value turns off live-plotting."""
        return self._refresh

    @refresh.setter
    def refresh(self, value: float):
        """ Set the refresh rate (per second). A negative refresh rate
            turns off static plotting."""
        assert isinstance(value, Real)
        # TODO: Proper input validation
        self._refresh = 0.001 if 0 <= value < 0.001 else value
        self._liveplot = self._refresh >= 0.0 and "nbAgg" in self._backend

    @property
    def figsize(self) -> Optional[Tuple[float, float]]:
        """Returns the current size of the figure in inches."""
        return self._pltkwargs["figsize"]

    @figsize.setter
    def figsize(self, size: Tuple[float, float]):
        if (
            not isinstance(size, Sequence)
            or len(size) != 2
            or not all(isinstance(x, Real) and x > 0 for x in size)
        ):
            raise ValueError(
                f"`size` must be a length-2 sequence of "
                f"positive-valued numbers, got: {size}"
            )
        size = tuple(size)
        if self.figsize != size:
            self._pltkwargs["figsize"] = size
            if self._fig is not None:
                self._fig.set_size_inches(size)

    @property
    def plot_objects(self) -> Union[Tuple[Figure, Axes], Tuple[Figure, np.ndarray]]:
        """ The figure-instance of the plot, and the axis-instance for each metric.

        Notes
        -----
        Calling this method will initialize the plot window if it is not already
        rendered.

        Returns
        -------
        Union[Tuple[Figure, Axes], Tuple[Figure, np.ndarray]]
            If more than one set of axes are present in the figure, an array of
            axes is returned instead."""
        if self._fig is None:
            self._init_plot_window()

        if self._axes.size == 1:
            return self._fig, self._axes.item()
        else:
            return self._fig, self._axes

    def __init__(
        self,
        metrics: Metrics,
        refresh: Real = 0.0,
        nrows: Optional[int] = None,
        ncols: int = 1,
        figsize: Optional[Tuple[int, int]] = None,
    ):
        """ Parameters
            ----------
            metrics : Union[str, Sequence[str], Dict[str, valid-color], Dict[str, Dict['train'/'test', valid-color]]]
                The name, or sequence of names, of the metric(s) that will be plotted.

            `metrics` can also be a dictionary, specifying the colors used to plot
            the metrics. Two mappings are valid:
                - '<metric-name>' -> color-value  (specifies train-metric color only)
                - '<metric-name>' -> {'train'/'test' : color-value}

            refresh : float, optional (default=0.)
                Sets the plot refresh rate in seconds.

                A refresh rate of 0. updates the once every 1/1000 seconds.

                A negative refresh rate  turns off live plotting:
                   Call `self.plot()` to draw the static plot.
                   Call `self.show()` to open a window showing the static plot

            nrows : Optional[int]
                Number of rows of the subplot grid. Metrics are added in
                row-major order to fill the grid.

            ncols : int, optional, default: 1
                Number of columns of the subplot grid. Metrics are added in
                row-major order to fill the grid.

            figsize : Optional[Sequence[int, int]]
                Specifies the width and height, respectively, of the figure."""
        # type checking on inputs
        # initializes the batch and epoch numbers
        super().__init__()

        assert isinstance(refresh, Real)
        assert (
            figsize is None
            or len(figsize) == 2
            and all(isinstance(i, Integral) for i in figsize)
        )

        # import matplotlib and check backend
        self._pyplot = importlib.import_module("matplotlib.pyplot")
        _matplotlib = importlib.import_module("matplotlib")

        self._backend = _matplotlib.get_backend()
        if "nbAgg" not in self._backend and refresh >= 0:
            _inline_msg = """Live plotting is not supported when matplotlib uses the '{}'
                             backend. Instead, use the 'nbAgg' backend.

                             In a Jupyter notebook, this can be activated using the cell magic:
                                %matplotlib notebook."""
            warn(cleandoc(_inline_msg.format(self._backend)))

        # input parameters
        self._metrics = (metrics,) if isinstance(metrics, str) else tuple(metrics)

        if not len(self._metrics) == len(set(self._metrics)):
            from collections import Counter

            count = Counter(self._metrics)
            _items = [name for name, cnt in count.most_common() if cnt > 1]
            raise ValueError(
                "`metrics` must specify mutually-unique names. "
                "\n `{}` {} specified redundantly".format(
                    ", ".join(_items), "was" if len(_items) == 1 else "were"
                )
            )

        if not self._metrics:
            raise ValueError("At least one metric must be specified")

        if any(not isinstance(i, str) for i in self._metrics):
            raise TypeError("`metrics` must be a string or a collection of strings")

        if nrows is None:
            nrows = 1

        if not isinstance(nrows, Integral) or 1 > nrows:
            raise ValueError(
                "`nrows` must integer-valued and be at least 1. Got {}".format(nrows)
            )

        if not isinstance(ncols, Integral) or 1 > ncols:
            raise ValueError(
                "`ncols` must integer-valued and be at least 1. Got {}".format(ncols)
            )

        if len(self._metrics) > ncols * nrows:
            nrows = len(self._metrics)

        self._refresh = None  # type: Optional[float]
        self._liveplot = None  # type: Optional[bool]
        self.refresh = refresh  # sets _refresh and _liveplot

        self._pltkwargs = dict(figsize=figsize, nrows=nrows, ncols=ncols)

        # color config
        self._train_colors = defaultdict(lambda: None)
        self._test_colors = defaultdict(lambda: None)
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, dict):
                    self._train_colors[k] = v.get("train")
                    self._test_colors[k] = v.get("test")
                else:
                    self._train_colors[k] = v
        sum(_check_valid_color(c) for c in self._train_colors.values())
        sum(_check_valid_color(c) for c in self._test_colors.values())

        # plot-settings for batch and epoch data
        self._batch_ax = dict(ls="-", alpha=0.5)
        self._epoch_ax = dict(ls="-", marker="o", markersize=6, lw=3)
        self._legend = dict()

        # metric name -> matplotlib axis object
        self._axis_mapping = OrderedDict()  # type: Dict[str, Axes]

        self._plot_batch = True  # type: bool
        self._fig = None  # type: Optional[Figure]
        self._axes = None  # type: Union[None, Axes, np.ndarray]

        self._last_plot_time = None  # type: Optional[float]

    def to_dict(self):
        out = super().to_dict()
        out.update(
            dict(
                refresh=self.refresh,
                pltkwargs=self._pltkwargs,
                train_colors=dict(self._train_colors),
                test_colors=dict(self._test_colors),
                metric_names=self._metrics,
            )
        )
        return out

    @classmethod
    def from_dict(cls, plotter_dict):
        new = cls(metrics=plotter_dict["metric_names"], refresh=plotter_dict["refresh"])

        new._train_metrics.update(
            (key, LiveMetric.from_dict(metric))
            for key, metric in plotter_dict["train_metrics"].items()
        )

        new._test_metrics.update(
            (key, LiveMetric.from_dict(metric))
            for key, metric in plotter_dict["test_metrics"].items()
        )

        for train_mode, stat_mode in product(["train", "test"], ["batch", "epoch"]):
            item = "num_{}_{}".format(train_mode, stat_mode)
            setattr(new, "_" + item, plotter_dict[item])

        for attr in ("pltkwargs", "train_colors", "test_colors"):
            setattr(new, "_" + attr, plotter_dict[attr])

        train_colors = defaultdict(lambda: None)
        test_colors = defaultdict(lambda: None)
        train_colors.update(new._train_colors)
        test_colors.update(new._test_colors)
        new._train_colors = train_colors
        new._test_colors = test_colors
        return new

    def set_train_batch(
        self, metrics: Dict[str, Real], batch_size: Integral, plot: bool = True
    ):
        """ Provide the batch-level metric values to be recorded, and (optionally) plotted.

            Parameters
            ----------
            metrics : Dict[str, Real]
                Mapping of metric-name to value. Only those metrics that were
                registered when initializing LivePlot will be recorded.

            batch_size : Integral
                The number of samples in the batch used to produce the metrics.
                Used to weight the metrics to produce epoch-level statistics.

            plot : bool
                If True, plot the batch-metrics (adhering to the refresh rate)"""

        if not self._num_train_batch:
            self._init_plot_window()

            unreg_metrics = set(metrics).difference(self._metrics)
            if unreg_metrics:
                msg = (
                    "\nThe following training metrics are not registered for live-plotting:\n\t"
                    + "\n\t"
                )
                warn(cleandoc(msg.join(sorted(unreg_metrics))))

        super().set_train_batch(
            {k: v for k, v in metrics.items() if k in self._metrics},
            batch_size=batch_size,
        )

        self._plot_batch = plot

        if self._plot_batch:
            self._do_liveplot()

    def plot_train_epoch(self):
        """
        Compute the epoch-level train statistics and plot the data point.
        """
        super().set_train_epoch()
        self._do_liveplot()

    def set_test_batch(self, metrics: Dict[str, Real], batch_size: Integral):
        """ Provide the batch-level metric values to be recorded, and (optionally) plotted.

            Parameters
            ----------
            metrics : Dict[str, Real]
                Mapping of metric-name to value. Only those metrics that were
                registered when initializing LivePlot will be recorded.

            batch_size : Integral
                The number of samples in the batch used to produce the metrics.
                Used to weight the metrics to produce epoch-level statistics.
            """
        # initialize live plot objects for testing
        if not self._test_metrics:
            unreg_metrics = set(metrics).difference(self._metrics)
            if unreg_metrics:
                msg = (
                    "\nThe following testing metrics are not registered for live-plotting:\n\t"
                    + "\n\t"
                )
                warn(cleandoc(msg + "\n\t".join(sorted(unreg_metrics))))

        super().set_test_batch(
            {k: v for k, v in metrics.items() if k in self._metrics},
            batch_size=batch_size,
        )

    def plot_test_epoch(self):
        """
        Compute the epoch-level test statistics and plot the data point.
        """
        if not self._num_test_epoch:
            self._init_plot_window()

        super().set_test_epoch()
        self._do_liveplot()

    def _init_plot_window(self):
        if self._fig is not None:
            return None

        self._fig, self._axes = self._pyplot.subplots(sharex=True, **self._pltkwargs)
        self._fig.tight_layout()

        self._pltkwargs["figsize"] = tuple(self._fig.get_size_inches())

        if len(self._metrics) == 1:
            self._axes = np.array([self._axes])

        # remove unused axes from plot grid
        axis_offset = self._axes.size - len(self._metrics)
        for i, ax in zip(range(axis_offset), self._axes.flat[::-1]):
            ax.remove()

        self._axis_mapping.update(zip(self._metrics, self._axes.flat))

        for ax in self._axes.flat:
            ax.grid(True)

        # Add x-label to bottom-plot for each column
        for i in range(min(self._pltkwargs["ncols"], len(self._metrics))):
            self._axes.flat[-(i + 1 + axis_offset)].set_xlabel("Number of iterations")

    def plot(self):
        """ Plot data, irrespective of the refresh rate. This should only
           be called if you are generating a static plot."""
        # plot batch-level train metrics
        for key, livedata in self._train_metrics.items():
            if livedata._batch_data and livedata.batch_line is None:
                try:
                    ax = self._axis_mapping[key]
                    livedata.batch_line, = ax.plot(
                        [],
                        [],
                        label="train",
                        color=self._train_colors.get(key),
                        **self._batch_ax,
                    )
                    ax.set_title(key)
                    ax.legend()
                except KeyError:
                    pass

            if self._plot_batch:
                livedata.batch_line.set_xdata(livedata.batch_domain)
                livedata.batch_line.set_ydata(livedata._batch_data)
                if livedata._epoch_data:
                    livedata.batch_line.set_label(
                        "train: {:.2e}".format(livedata._epoch_data[-1])
                    )

        # plot epoch-level train metrics
        for key, livedata in self._train_metrics.items():
            if livedata.epoch_line is None:
                # initialize batch-level plot objects
                ax = self._axis_mapping[key]
                batch_color = self._train_metrics[key].batch_line.get_color()
                livedata.epoch_line, = ax.plot(
                    [], [], color=batch_color, **self._epoch_ax
                )
                ax.legend(**self._legend)

            if livedata.epoch_line is not None:
                livedata.epoch_line.set_xdata(livedata._epoch_domain)
                livedata.epoch_line.set_ydata(livedata._epoch_data)

        # plot epoch-level test metrics
        for key, livedata in self._test_metrics.items():
            # initialize epoch-level plot objects
            if livedata.epoch_line is None:
                try:
                    ax = self._axis_mapping[key]
                    livedata.epoch_line, = ax.plot(
                        [],
                        [],
                        label="test",
                        color=self._test_colors.get(key),
                        **self._epoch_ax,
                    )
                    ax.set_title(key)
                    ax.legend(**self._legend)
                except KeyError:
                    pass

            if livedata.epoch_line is not None:
                livedata.epoch_line.set_xdata(livedata._epoch_domain)
                livedata.epoch_line.set_ydata(livedata._epoch_data)
                if livedata._epoch_data:
                    livedata.epoch_line.set_label(
                        "test: " + "{:.2e}".format(livedata._epoch_data[-1])
                    )

        self._update_text()
        self._resize()
        if self._liveplot:
            self._fig.canvas.draw()

        self._last_plot_time = time.time()

    def _resize(self):
        if self._axes is None:
            return

        for ax in self._axes.flat:
            ax.relim()
            ax.autoscale_view()

    def _update_text(self):
        for ax in self._axis_mapping.values():
            ax.legend()

    def show(self):
        """ Calls `matplotlib.pyplot.show()`. For visualizing a static-plot"""
        if not self._liveplot:
            self._pyplot.show()

    def _do_liveplot(self):
        # enable active plotting upon first plot
        if self._last_plot_time is None:
            if self._liveplot:
                self._pyplot.ion()
            self._last_plot_time = time.time()

        if self._liveplot and time.time() - self._last_plot_time >= self._refresh:
            self.plot()

    def set_test_epoch(self):
        """ Not implemented. Use `plot_test_epoch` instead"""
        raise NotImplementedError("Use the method `plot_test_epoch` instead")

    def set_train_epoch(self):
        """ Not implemented. Use `plot_train_epoch` instead"""
        raise NotImplementedError("Use the method `plot_train_epoch` instead")
