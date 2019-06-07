from collections import OrderedDict, defaultdict, namedtuple
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from custom_inherit import doc_inherit

from noggin.logger import LiveLogger
from noggin.plotter import LivePlot
from noggin.typing import Axes, Figure, LiveMetrics, Metrics, ValidColor, ndarray

__all__ = ["create_plot", "save_metrics", "load_metrics"]


@doc_inherit(LivePlot.__init__, style="numpy")
def create_plot(
    metrics: Metrics,
    max_fraction_spent_plotting: float = 0.05,
    last_n_batches: Optional[int] = None,
    nrows: Optional[int] = None,
    ncols: int = 1,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[LivePlot, Figure, ndarray]:
    """ Create matplotlib figure/axes, and a live-plotter, which publishes
    "live" training/testing metric data, at a batch and epoch level, to
    the figure.

    Returns
    -------
    Tuple[liveplot.LivePlot, matplotlib.figure.Figure, numpy.ndarray(matplotlib.axes.Axes)]
        (LivePlot-instance, figure, array-of-axes)


    Examples
    --------
    Creating a live plot in a Jupyter notebook

    >>> %matplotlib notebook
    >>> import numpy as np
    >>> from noggin import create_plot, save_metrics
    >>> metrics = ["accuracy", "loss"]
    >>> plotter, fig, ax = create_plot(metrics)
    >>> for i, x in enumerate(np.linspace(0, 10, 100)):
    ...     # training
    ...     x += np.random.rand(1)*5
    ...     batch_metrics = {"accuracy": x**2, "loss": 1/x**.5}
    ...     plotter.set_train_batch(batch_metrics, batch_size=1, plot=True)
    ...
    ...     # cue training epoch
    ...     if i%10 == 0 and i > 0:
    ...         plotter.plot_train_epoch()
    ...
    ...         # cue test-time computations
    ...         for x in np.linspace(0, 10, 5):
    ...             x += (np.random.rand(1) - 0.5)*5
    ...             test_metrics = {"accuracy": x**2}
    ...             plotter.set_test_batch(test_metrics, batch_size=1)
    ...         plotter.plot_test_epoch()
    ...
    ... plotter.plot()  # ensures final data gets plotted

    Saving the logged metrics

    >>> save_metrics("./metrics.npz", plotter) # save metrics to numpy-archive
    """
    live_plotter = LivePlot(
        metrics,
        max_fraction_spent_plotting=max_fraction_spent_plotting,
        last_n_batches=last_n_batches,
        figsize=figsize,
        ncols=ncols,
        nrows=nrows,
    )
    fig, ax = live_plotter.plot_objects
    return live_plotter, fig, ax


def save_metrics(
    path: Union[str, Path],
    liveplot: Optional[Union[LivePlot, LiveLogger]] = None,
    *,
    train_metrics: LiveMetrics = None,
    test_metrics: LiveMetrics = None
):
    """ Save live-plot metrics to a numpy zipped-archive (.npz). A LivePlot-instance
        can be supplied, or train/test metrics can be passed explicitly to the function.

        Parameters
        ----------
        path: PathLike
           The file-path used to save the archive. E.g. 'path/to/saved_metrics.npz'

        liveplot : Optional[noggin.LivePlot]
           The LivePlot instance whose metrics will be saves.

        train_metrics : Optional[OrderedDict[str, Dict[str, numpy.ndarray]]]]

            '<metric-name>' -> {'batch_data'   -> array,
                                'epoch_domain' -> array,
                                'epoch_data'   -> array}

        test_metrics : Optional[OrderedDict[str, Dict[str, numpy.ndarray]]]]

            '<metric-name>' -> {'batch_data'   -> array,
                                'epoch_domain' -> array,
                                'epoch_data'   -> array}"""
    if liveplot is not None:
        train_metrics = liveplot.train_metrics
        test_metrics = liveplot.test_metrics
    else:
        if train_metrics is None:
            train_metrics = {}

        if test_metrics is None:
            test_metrics = {}

    # use unique separator
    sep = ";"
    names = "".join(tuple(train_metrics) + tuple(test_metrics))
    while sep in names:
        sep += ";"

    # flatten metrics to single mapping
    save_dict = {}  # train/test;metric_name;metric_data -> array
    for type_, metrics in zip(["train", "test"], [train_metrics, test_metrics]):
        for name, metric in metrics.items():
            save_dict.update({sep.join((type_, name, k)): v for k, v in metric.items()})

    with open(path, "wb") as f:
        np.savez(
            f,
            train_order=list(train_metrics),
            test_order=list(test_metrics),
            sep=sep,
            **save_dict
        )


metrics = namedtuple("metrics", ["train", "test"])


def load_metrics(path: Union[str, Path]) -> Tuple[LiveMetrics, LiveMetrics]:
    """ Load noggin metrics from a numpy archive.

        Parameters
        ----------
        path : PathLike
            Path to numpy archive.

        Returns
        -------
        Tuple[OrderedDict[str, Dict[str, numpy.ndarray]], OrderedDict[str, Dict[str, numpy.ndarray]]]
           (train-metrics, test-metrics)"""

    def recursive_default_dict():
        return defaultdict(recursive_default_dict)

    out = recursive_default_dict()

    with np.load(path) as f:
        data_dict = dict(f)

    train_order = list(data_dict.pop("train_order"))
    test_order = list(data_dict.pop("test_order"))
    sep = data_dict.pop("sep").item()
    for k, v in data_dict.items():
        if v.ndim == 0:
            v = v.item()
        type_, metric_name, data_type = k.split(sep)
        out[type_][metric_name][data_type] = v

    train_metrics = OrderedDict(((k, dict(out["train"][k])) for k in train_order))
    test_metrics = OrderedDict(((k, dict(out["test"][k])) for k in test_order))
    return metrics(train_metrics, test_metrics)


def plot_logger(
    logger: LiveLogger,
    plot_batches: bool = True,
    last_n_batches: Optional[int] = None,
    colors: Optional[Dict[str, Union[ValidColor, Dict[str, ValidColor]]]] = None,
) -> Tuple[LivePlot, Figure, Union[Axes, np.ndarray]]:
    """Plots the data recorded by a :class:`~noggin.plotter.LiveLogger` instance.

    Converts the logger to an instance of :class:`~noggin.plotter.LivePlot`.

    Parameters
    ----------
    logger : LiveLogger
        The logger whose train/test-split batch/epoch-level data will be plotted.

    plot_batches : bool, optional (default=True)
        If ``True`` include batch-level data in plot.

    last_n_batches : Optional[int]
        The maximum number of batches to be plotted at any given time.
        If ``None``, all of the data will be plotted.

    colors : Optional[Dict[str, Union[ValidColor, Dict[str, ValidColor]]]]
        ``colors`` can be a dictionary, specifying the colors used to plot
        the metrics. Two mappings are valid:
            - '<metric-name>' -> color-value  (specifies train-metric color only)
            - '<metric-name>' -> {'train'/'test' : color-value}
        If ``None``, default colors are used in the plot.

    Returns
    -----
    Tuple[LivePlot, Figure, Union[Axes, np.ndarray]]
        The resulting plotter, matplotlib-figure, and axis (or array of axes)
    """

    if not isinstance(logger, LiveLogger):
        raise TypeError(
            "`logger` must be an instance of `noggin.LiveLogger`, got {}".format(logger)
        )

    metrics = sorted(
        set(list(logger.train_metrics.keys()) + list(logger.test_metrics.keys()))
    )

    plotter = LivePlot(
        metrics, max_fraction_spent_plotting=0.0, last_n_batches=last_n_batches
    )

    plotter.last_n_batches = last_n_batches
    if colors is not None:
        plotter.metric_colors = colors

    plotter_dict = plotter.to_dict()

    plotter_dict.update(logger.to_dict())
    plotter = LivePlot.from_dict(plotter_dict)
    plotter.plot(plot_batches=plot_batches)
    fig, ax = plotter.plot_objects
    return plotter, fig, ax
