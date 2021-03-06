from math import ceil
import importlib
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Sequence
from inspect import cleandoc
from itertools import product
from numbers import Integral, Real
from typing import Dict, Optional, Set, Tuple, Union
from warnings import warn

import numpy as np
from matplotlib.pyplot import Axes, Figure

from noggin.logger import LiveLogger, LiveMetric
from noggin.typing import Metrics, ValidColor

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


class LivePlot(LiveLogger):
    """Records and plots batch-level and epoch-level summary statistics of the training and
    testing metrics of a model during a session.

    The rate at which the plot is updated is controlled by
    :obj:`~noggin.plotter.LivePlot.max_fraction_spent_plotting`.

    The maximum number of batches to be included in the plot is controlled by
    :obj:`~noggin.plotter.LivePlot.last_n_batches`.

    Notes
    -----
    Live plotting is only supported for the 'nbAgg' backend (i.e.
    when the cell magic ``%matplotlib notebook`` is invoked in a
    jupyter notebook). """

    @property
    def metrics(self) -> Tuple[str, ...]:
        """A tuple of all the metric names"""
        return self._metrics

    @property
    def metric_colors(self) -> Dict[str, Dict[str, ValidColor]]:
        """The color associated with each of the train/test and batch/epoch-level
        metrics.

        Returns
        -------
        Dict[str, Dict[str, color-value]]
            {'<metric-name>' -> {'train'/'test' -> color-value}}"""
        out = defaultdict(dict)
        for k, v in self._train_colors.items():
            out[k]["train"] = v

        for k, v in self._test_colors.items():
            out[k]["test"] = v
        return dict(out)

    @metric_colors.setter
    def metric_colors(self, value: Dict[str, Union[ValidColor, Dict[str, ValidColor]]]):
        if not isinstance(value, dict):
            raise TypeError(
                "`metric_colors` must be a dictionary that maps:"
                "\nmetric-name -> valid-color"
                "\nor"
                "\nmetric-name -> 'train' -> valid-color"
                "\n               'test'  -> valid-color"
                "\nGot: {}".format(value)
            )
        for k, v in value.items():
            if k not in self.metrics:
                continue
            if isinstance(v, dict):
                self._train_colors[k] = v.get("train")
                self._test_colors[k] = v.get("test")
            else:
                self._train_colors[k] = v
        sum(_check_valid_color(c) for c in self._train_colors.values())
        sum(_check_valid_color(c) for c in self._test_colors.values())

    @property
    def figsize(self) -> Optional[Tuple[float, float]]:
        """Returns the current size of the figure in inches.

        Parameters
        ----------
        Optional[Tuple[float, float]]
        """
        return self._pltkwargs.get("figsize")

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
            axis = self._axes.item()  # type: Axes
            return self._fig, axis
        else:
            return self._fig, self._axes

    @property
    def max_fraction_spent_plotting(self) -> float:
        """The maximum fraction of time spent plotting.

        Parameters
        ----------
        value : float
            A value in [0, 1]. A value of ``0.0`` turns live-plotting off.
            A value of ``1.0`` will result in the plot updating whenever a
            new measurement is recorded.

        Notes
        -----
        The refresh rate for plotting will update dynamically such that::

            mean_plot_time / (time_since_last_plot + mean_plot_time)

        does not exceed ``max_fraction_spent_plotting``."""
        return self._max_fraction_spent_plotting

    @max_fraction_spent_plotting.setter
    def max_fraction_spent_plotting(self, value: float):
        if not isinstance(value, (int, float)):
            raise TypeError(
                "`max_fraction_spent_plotting` must be a "
                "floating point number in [0, 1], got {}".format(value)
            )

        if not 0 <= value <= 1:
            raise ValueError(
                "`max_fraction_spent_plotting` must be a "
                "floating point number in [0, 1], got {}".format(value)
            )
        self._max_fraction_spent_plotting = value

    @property
    def last_n_batches(self) -> int:
        """The maximum number of batches to be plotted at any given time.
        If ``None``, all data will be plotted.

        Parameters
        ----------
        value : Union[int, None]
        """
        return self._last_n_batches

    @last_n_batches.setter
    def last_n_batches(self, value: int):
        self._epoch_domain_lookup = dict(train=defaultdict(int), test=defaultdict(int))
        # type: Dict[str, Dict[str, int]]

        if value is None:
            self._last_n_batches = None
            return

        if not isinstance(value, int):
            raise TypeError(
                "`last_n_batches` must be a positive integer, got {}".format(value)
            )

        if value < 1:
            raise ValueError(
                "`last_n_batches` must be a positive integer, got {}".format(value)
            )
        # Points to starting index for the epoch-domain of a
        # given metric's name; used to keep epoch plot within
        # "last-n-batches" plotted.
        # This must be reset each time `last_n_batches` is set
        self._last_n_batches = value

    def __init__(
        self,
        metrics: Metrics,
        max_fraction_spent_plotting: float = 0.05,
        last_n_batches: Optional[int] = None,
        nrows: Optional[int] = None,
        ncols: int = 1,
        figsize: Optional[Tuple[int, int]] = None,
    ):
        """
        Parameters
        ----------
        metrics : Union[str, Sequence[str], Dict[str, valid-color], Dict[str, Dict['train'/'test', valid-color]]]
            The name, or sequence of names, of the metric(s) that will be plotted.

            ``metrics`` can also be a dictionary, specifying the colors used to plot
            the metrics. Two mappings are valid:
                - '<metric-name>' -> color-value  (specifies train-metric color only)
                - '<metric-name>' -> {'train'/'test' : color-value}

        max_fraction_spent_plotting : float, optional (default=0.05)
            The maximum fraction of time spent plotting. The default value is ``0.05``,
            meaning that no more than 5% of processing time will be spent plotting, on
            average.

        last_n_batches : Optional[int]
            The maximum number of batches to be plotted at any given time.
            If ``None``, all data will be plotted.

        nrows : Optional[int]
            Number of rows of the subplot grid. Metrics are added in
            row-major order to fill the grid.

        ncols : int, optional, default: 1
            Number of columns of the subplot grid. Metrics are added in
            row-major order to fill the grid.

        figsize : Optional[Sequence[float, float]]
            Specifies the width and height, respectively, of the figure."""
        # type checking on inputs
        # initializes the batch and epoch numbers
        super().__init__()

        # import matplotlib and check backend
        self._pyplot = importlib.import_module("matplotlib.pyplot")
        _matplotlib = importlib.import_module("matplotlib")

        self._backend = _matplotlib.get_backend()
        self._liveplot = "nbAgg" in self._backend

        # plot-settings for batch and epoch data
        self._batch_ax = dict(ls="-", alpha=0.5)
        self._epoch_ax = dict(ls="-", marker="o", markersize=6, lw=3)
        self._legend = dict()

        # metric name -> matplotlib axis object
        self._axis_mapping = OrderedDict()  # type: Dict[str, Axes]

        # plot objects
        self._fig = None  # type: Optional[Figure]
        self._axes = None  # type: Union[None, Axes, np.ndarray]

        # plotting logic
        self._plot_batch = True  # type: bool
        self._last_plot_time = None  # type: Optional[float]
        self._plot_time_queue = deque([])  # stores most recent plot-times (seconds)
        self._time_of_last_liveplot_attempt = None  # type: Optional[float]
        self._draw_time = 0.0  # type: float

        # 'train/test' -> {metric-name -> batch-index of most-recent epoch}
        self._epoch_domain_lookup = dict(train=defaultdict(int), test=defaultdict(int))
        # type: Dict[str, Dict[str, int]]

        self.last_n_batches = last_n_batches

        # used to warn users only once when they plot an unregistered metric
        self._unregistered_metrics = set()  # type: Set[str]

        # stores most times between consecutive live-plot attempts (seconds)
        self._queue_size = 4
        self.max_fraction_spent_plotting = max_fraction_spent_plotting

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
            nrows = int(ceil(len(self._metrics) / ncols))

        assert nrows * ncols >= len(self._metrics)

        self._pltkwargs = dict(nrows=nrows, ncols=ncols)
        if figsize is not None:
            self.figsize = figsize
        else:
            self._pltkwargs["figsize"] = None

        # color config
        self._train_colors = defaultdict(lambda: None)
        self._test_colors = defaultdict(lambda: None)
        if isinstance(metrics, dict):
            self.metric_colors = metrics

        if "nbAgg" not in self._backend and max_fraction_spent_plotting > 0.0:
            _inline_msg = """Live plotting is not supported when matplotlib uses the '{}'
                             backend. Instead, use the 'nbAgg' backend.

                             In a Jupyter notebook, this can be activated using the cell magic:
                                %matplotlib notebook."""
            warn(cleandoc(_inline_msg.format(self._backend)))

    def to_dict(self):
        """Records the state of the plotter in a dictionary.

        This is the inverse of :func:`~noggin.plotter.LivePlot.from_dict`

        Returns
        -------
        Dict[str, Any]

        Notes
        -----
        To save your plotter, use this method to convert it to a dictionary
        and then pickle the dictionary.
        """
        out = super().to_dict()
        out.update(
            dict(
                max_fraction_spent_plotting=self.max_fraction_spent_plotting,
                last_n_batches=self.last_n_batches,
                pltkwargs=self._pltkwargs,
                train_colors=dict(self._train_colors),
                test_colors=dict(self._test_colors),
                metric_names=self._metrics,
            )
        )
        return out

    @classmethod
    def from_dict(cls, plotter_dict):
        """Records the state of the plotter in a dictionary.

        This is the inverse of :func:`~noggin.plotter.LivePlot.to_dict`

        Parameters
        ----------
        plotter_dict : Dict[str, Any]
            The dictionary storing the state of the logger to be
            restored.

        Returns
        -------
        noggin.LivePlot
            The restored plotter.

        Notes
        -----
        This is a class-method, the syntax for invoking it is:

        >>> loaded_plotter = LivePlot.from_dict(plotter_dict)

        To restore your plot from the loaded plotter, call:

        >>> loaded_plotter.plot()
        """
        new = cls(
            metrics=plotter_dict["metric_names"],
            max_fraction_spent_plotting=plotter_dict["max_fraction_spent_plotting"],
            last_n_batches=plotter_dict["last_n_batches"],
        )

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

    def _filter_unregistered_metrics(self, metrics: Dict[str, Real]) -> Dict[str, Real]:
        """
        Returns
        -------
        Dict[str, Real]
            A dictionary containing only registered metric-names.

        Warns
        -----
        UserWarning
            Unknown metric was logged
        """
        unknown_metrics = set(metrics).difference(self._metrics)
        if unknown_metrics - self._unregistered_metrics:
            msg = "\nThe following metrics are not registered for live-plotting: "
            warn(msg + ", ".join(sorted(unknown_metrics - self._unregistered_metrics)))
        self._unregistered_metrics.update(unknown_metrics)
        return (
            {k: v for k, v in metrics.items() if k in self._metrics}
            if unknown_metrics
            else metrics
        )

    def set_train_batch(
        self, metrics: Dict[str, Real], batch_size: Integral, plot: bool = True
    ):
        """Record batch-level measurements for train-metrics, and (optionally)
        plot them.

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

        super().set_train_batch(
            self._filter_unregistered_metrics(metrics), batch_size=batch_size
        )

        self._plot_batch = plot

        if self._plot_batch:
            self._do_liveplot()

    def set_train_epoch(self):
        """Record and plot an epoch for the train-metrics.

        Computes epoch-level statistics based on the batches accumulated since
        the prior epoch.
        """
        super().set_train_epoch()
        self._do_liveplot()

    def set_test_batch(self, metrics: Dict[str, Real], batch_size: Integral):
        """Record batch-level measurements for test-metrics.

        Parameters
        ----------
        metrics : Dict[str, Real]
            Mapping of metric-name to value. Only those metrics that were
            registered when initializing LivePlot will be recorded.

        batch_size : Integral
            The number of samples in the batch used to produce the metrics.
            Used to weight the metrics to produce epoch-level statistics."""
        super().set_test_batch(
            self._filter_unregistered_metrics(metrics), batch_size=batch_size
        )

    def set_test_epoch(self):
        """Record and plot an epoch for the test-metrics.

        Computes epoch-level statistics based on the batches accumulated since
        the prior epoch.
        """
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

        self._pyplot.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    def plot(self, plot_batches: bool = True):
        """Plot the logged data.

        This method can be used to 'force' a plot to be drawn, and should *not* be
        called repeatedly while logging data.

        Instead, one should invoke ``Liveplot.set_train_batch(plot=True)``,
        ``Liveplot.set_train_epoch``, and ``Liveplot.set_test_epoch``, which will
        adjust their plot-rates according to ``Liveplot.max_fraction_spent_plotting``.

        ``LivePlot.plot`` should be called at the end of a logging-loop to ensure that
        the logged data is plotted in its entirety. This can also be used to recreate
        a plot after deserializing a ``LivePlot`` instance.

        Parameters
        ----------
        plot_batches : bool, optional (default=True)
            If ``True`` include batch-level data in plot."""
        if not isinstance(plot_batches, bool):
            raise TypeError(
                "`plot_batch` must be `True` or `False`, got {}".format(plot_batches)
            )

        self._init_plot_window()
        for key, livedata in self._train_metrics.items():
            if livedata.batch_line is None:
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

            if plot_batches and livedata.batch_line:
                n = (
                    self.last_n_batches
                    if self.last_n_batches
                    else len(livedata.batch_domain)
                )
                livedata.batch_line.set_xdata(livedata.batch_domain[-n:])
                livedata.batch_line.set_ydata(livedata.batch_data[-n:])
                if livedata.epoch_data.size:
                    livedata.batch_line.set_label(
                        "train: {:.2e}".format(livedata.epoch_data[-1])
                    )
            elif len(livedata.batch_line.get_xdata()):
                # clear batch-level plots
                livedata.batch_line.set_xdata(np.array([]))
                livedata.batch_line.set_ydata(np.array([]))

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

            if livedata.epoch_line is not None and livedata.batch_domain.size:
                self._update_epoch_domain(
                    self.last_n_batches,
                    batch_domain=livedata.batch_domain,
                    epoch_domain_lookup=self._epoch_domain_lookup["train"],
                    livedata=livedata,
                )

        # plot epoch-level test metrics
        for key, livedata in self._test_metrics.items():
            # initialize epoch-level plot objects
            if livedata.epoch_line is None:
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

            if livedata.epoch_line is not None and livedata.batch_domain.size:
                if (
                    livedata.name in self._train_metrics
                    and self._train_metrics[livedata.name].batch_domain.size
                ):
                    batch_domain = self._train_metrics[livedata.name].batch_domain
                else:
                    batch_domain = livedata.batch_domain
                self._update_epoch_domain(
                    last_n_batches=self.last_n_batches,
                    batch_domain=batch_domain,
                    epoch_domain_lookup=self._epoch_domain_lookup["test"],
                    livedata=livedata,
                )
                if livedata.epoch_data.size:
                    livedata.epoch_line.set_label(
                        "test: " + "{:.2e}".format(livedata.epoch_data[-1])
                    )

        s = time.time()
        self._update_text()
        self._resize()
        if self._liveplot and self._fig is not None:
            self._fig.canvas.draw()
        self._draw_time = time.time() - s

    @staticmethod
    def _update_epoch_domain(
        last_n_batches: int,
        batch_domain: np.ndarray,
        epoch_domain_lookup: Dict[str, int],
        livedata: LiveMetric,
    ):
        """ Finds the oldest epoch batch-iteration within `last_n_batches` and
        sets the epoch-data such that it satisfies that bound.

        Parameters
        ----------
        last_n_batches : int

        batch_domain : numpy.ndarray
            The training batch data

        epoch_domain_lookup : Dict[str, int]
            metric-name -> batch-iteration of previous earliest-plotted-epoch

        livedata : LiveMetric
            The metric being updated
        """

        if last_n_batches:
            old_n = epoch_domain_lookup[livedata.name]
            n = np.searchsorted(
                livedata.epoch_domain[old_n:], batch_domain[-last_n_batches:][0]
            )
            n += old_n
        else:
            n = 0
        epoch_domain_lookup[livedata.name] = n
        livedata.epoch_line.set_xdata(livedata.epoch_domain[n:])
        livedata.epoch_line.set_ydata(livedata.epoch_data[n:])

    def _timed_plot(self, plot_batches: bool):
        plot_start_time = time.time()
        self.plot(plot_batches=plot_batches)
        self._last_plot_time = time.time()
        if len(self._plot_time_queue) == self._queue_size:
            self._plot_time_queue.popleft()
        self._plot_time_queue.append(self._last_plot_time - plot_start_time)

    def _do_liveplot(self):
        # enable active plotting upon first plot
        if self._last_plot_time is None:
            if self._liveplot:
                self._pyplot.ion()
            self._last_plot_time = time.time()

        if not self._liveplot:
            return

        self._time_of_last_liveplot_attempt = time.time()
        time_since_last_plot = (
            self._time_of_last_liveplot_attempt - self._last_plot_time
        )

        mean_plot_time = (
            sum(self._plot_time_queue) / len(self._plot_time_queue)
            if self._plot_time_queue
            else 0.0
        )

        if self.max_fraction_spent_plotting == 1.0 or (
            time_since_last_plot
            and mean_plot_time / (time_since_last_plot + mean_plot_time)
            < self.max_fraction_spent_plotting
        ):
            self._timed_plot(plot_batches=self._plot_batch)
            # exclude plot time
            self._time_of_last_liveplot_attempt = time.time()

    def _resize(self):
        if self._axes is None:  # pragma: no cover
            return

        for ax in self._axes.flat:
            ax.relim()
            ax.autoscale_view()

    def _update_text(self):
        for ax in self._axis_mapping.values():
            ax.legend()

    def show(self):  # pragma: no cover
        """ Calls ``matplotlib.pyplot.show()``. For visualizing a static-plot"""
        if not self._liveplot:
            self._pyplot.show()
