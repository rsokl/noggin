__all__ = ["create_plot", "recreate_plot", "save_metrics", "load_metrics"]


def create_plot(metrics, refresh=0., figsize=None, ncols=1, nrows=1):
    """ Create matplotlib figure/axes, and a live-plotter, which publishes
        "live" training/testing metric data, at a batch and epoch level, to
        the figure.

        Parameters
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

        figsize : Optional[Sequence[int, int]]
            Specifies the width and height, respectively, of the figure.

        nrows, ncols : int, optional, default: 1
            Number of rows/columns of the subplot grid. Metrics are added in
            row-major order to fill the grid.

        Returns
        -------
        Tuple[liveplot.LivePlot, matplotlib.figure.Figure, numpy.ndarray(matplotlib.axes.Axes)]
            (LivePlot-instance, figure, array-of-axes)


        Examples
        --------
        Creating a live plot in a Jupyter notebook
        >>> %matplotlib notebook
        >>> import numpy as np
        >>> from liveplot import create_plot, save_metrics
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
        >>> save_metrics("./metrics.npz", plotter) # save metrics to numpy-archive

        Loading and recreating plot
        >>> from liveplot import load_metrics, recreate_plot
        >>> train, test = load_metrics("./metrics.npz")
        >>> recreate_plot(train_metrics=train, test_metrics=test)"""

    from liveplot.plotter import LivePlot
    live_plotter = LivePlot(metrics, refresh, figsize=figsize, ncols=ncols, nrows=nrows)
    live_plotter._init_plot_window()
    fig, ax = live_plotter.plot_objects()
    return live_plotter, fig, ax


def recreate_plot(liveplot=None, *, train_metrics=None, test_metrics=None, colors=None, ncols=1, nrows=1):
    """ Recreate a plot from a LivePlot instance or from train/test metric dictionaries.

        Parameters
        ----------
        liveplot : Optional[liveplot.LivePlot]
            An existing liveplot object.

        Keyword-Only Arguments
        ----------------------
        train_metrics : Optional[OrderedDict[str, Dict[str, numpy.ndarray]]]
           '<metric-name>' -> {"batch_data":   array,
                               "epoch_data":   array,
                               "epoch_domain": array}

        test_metrics : Optional[OrderedDict[str, Dict[str, numpy.ndarray]]]
           '<metric-name>' -> {"batch_data":   array,
                               "epoch_data":   array,
                               "epoch_domain": array}

        colors : Optional[Dict[str, color-value], Dict[str, Dict[str, color-value]]
            Specifying train-time metric colors only:
                  metric-name -> color-value

            Specifying train or test-time metric colors:
                 metric-name -> {train/test -> color-value}

        nrows, ncols : int, optional, default: 1
            Number of rows/columns of the subplot grid. Metrics are added in
            row-major order to fill the grid.

        Returns
        -------
        Tuple[liveplot.LivePlot, matplotlib.figure.Figure, numpy.ndarray(matplotlib.axes.Axes)]
            (LivePlot-instance, figure, array-of-axes)"""
    from liveplot.plotter import LiveMetric
    from collections import defaultdict
    from liveplot.utils import check_valid_color

    assert any(i is not None for i in (liveplot, train_metrics, test_metrics))

    if liveplot is not None:
        assert train_metrics is None and test_metrics is None
    else:
        if train_metrics is None:
            train_metrics = {}
        if test_metrics is None:
            test_metrics = {}

    if liveplot:
        metrics = liveplot._metrics
    else:
        metrics = list(train_metrics)
        metrics.extend(k for k in test_metrics if k not in metrics)

    new, fig, ax = create_plot(metrics, refresh=-1, nrows=nrows, ncols=ncols)

    if liveplot:
        new._train_colors = liveplot._train_colors
        new._test_colors = liveplot._test_colors
    else:
        # set plotting colors
        train_colors = defaultdict(lambda: None)
        test_colors = defaultdict(lambda: None)
        if isinstance(colors, dict):
            for k, v in colors.items():
                if isinstance(v, dict):
                    train_colors[k] = v.get("train")
                    test_colors[k] = v.get("test")
                else:
                    train_colors[k] = v
        sum(check_valid_color(c) for c in train_colors.values())
        sum(check_valid_color(c) for c in test_colors.values())

        new._train_colors = train_colors
        new._test_colors = test_colors

    new._init_plot_window()
    fig, ax = new.plot_objects()

    if liveplot:
        new._train_metrics = liveplot._train_metrics
        new._test_metrics = liveplot._test_metrics

        for attr in ["_train_epoch_num", "_train_batch_num",
                     "_test_epoch_num", "_test_batch_num"]:
            setattr(new, attr, getattr(liveplot, attr))
    else:
        # initializing LiveMetrics and setting data
        new._train_metrics.update((key, LiveMetric(key)) for key in train_metrics if key in new._metrics)
        new._test_metrics.update((key, LiveMetric(key)) for key in test_metrics if key in new._metrics)

        for metric_name, metric in new._train_metrics.items():
            for item in ["batch_data", "epoch_data", "epoch_domain"]:
                setattr(metric, "_" + item, list(train_metrics[metric_name][item]))

        for metric_name, metric in new._test_metrics.items():
            for item in ["batch_data", "epoch_data", "epoch_domain"]:
                setattr(metric, "_" + item, list(test_metrics[metric_name][item]))

        # setting num_batch/num_epoch values
        num_name = ["_train_epoch_num", "_train_batch_num", "_test_epoch_num", "_test_batch_num"]
        vals = [0, 0, 0, 0]

        if train_metrics:
            vals[0] = max(len(v["epoch_data"]) for v in train_metrics.values())
            vals[1] = max(len(v["batch_data"]) for v in train_metrics.values())
        if test_metrics:
            vals[2] = max(len(v["epoch_data"]) for v in test_metrics.values())
            vals[3] = max(len(v["batch_data"]) for v in test_metrics.values())

        for name, val in zip(num_name, vals):
            setattr(new, name, val)

    # initialize train metric batch lines
    for key, metric in new._train_metrics.items():
        try:
            ax = new._axis_mapping[key]
            metric.batch_line, = ax.plot([], [], label="train", color=new._train_colors[key], **new._batch_ax)
            ax.set_ylabel(key)
            ax.legend()
        except KeyError:
            pass

    # initialize train metric epoch lines
    for key in new._train_metrics:
        ax = new._axis_mapping[key]
        batch_color = new._train_metrics[key].batch_line.get_color()
        new._train_metrics[key].epoch_line, = ax.plot([], [], color=batch_color, **new._epoch_ax)
        ax.legend(**new._legend)

    # initialize test metric epoch lines
    for key, metric in new._test_metrics.items():
        try:
            ax = new._axis_mapping[key]
            metric.epoch_line, = ax.plot([], [], label="test", color=new._test_colors[key], **new._epoch_ax)
            ax.set_ylabel(key)
            ax.legend(**new._legend)
        except KeyError:
            pass

    new.plot()
    return new, fig, ax


def save_metrics(path, liveplot=None, *, train_metrics=None, test_metrics=None):
    """ Save live-plot metrics to a numpy zipped-archive (.npz). A LivePlot-instance
        can be supplied, or train/test metrics can be passed explicitly to the function.

        Parameters
        ----------
        path: PathLike
           The file-path used to save the archive. E.g. 'path/to/saved_metrics.npz'

        liveplot : Optional[liveplot.LivePlot]
           The LivePlot instance whose metrics will be saves.

        train_metrics : Optional[OrderedDict[str, Dict[str, numpy.ndarray]]]]

            '<metric-name>' -> {'batch_data'   -> array,
                                'epoch_domain' -> array,
                                'epoch_data'   -> array}

        test_metrics : Optional[OrderedDict[str, Dict[str, numpy.ndarray]]]]

            '<metric-name>' -> {'batch_data'   -> array,
                                'epoch_domain' -> array,
                                'epoch_data'   -> array}"""
    import numpy as np

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
        np.savez(f, train_order=list(train_metrics), test_order=list(test_metrics),
                 sep=sep, **save_dict)


def load_metrics(path):
    """ Load liveplot metrics from a numpy archive.

        Parameters
        ----------
        path : PathLike
            Path to numpy archive.

        Returns
        -------
        Tuple[OrderedDict[str, Dict[str, numpy.ndarray]], OrderedDict[str, Dict[str, numpy.ndarray]]]]
           (train-metrics, test-metrics)"""
    from collections import OrderedDict, defaultdict
    import numpy as np

    def recursive_default_dict(): return defaultdict(recursive_default_dict)
    out = recursive_default_dict()

    with np.load(path) as f:
        data_dict = dict(f)

    train_order = list(data_dict.pop("train_order"))
    test_order = list(data_dict.pop("test_order"))
    sep = data_dict.pop("sep").item()
    for k, v in data_dict.items():
        type_, metric_name, data_type = k.split(sep)
        out[type_][metric_name][data_type] = v

    train_metrics = OrderedDict(((k, dict(out["train"][k])) for k in train_order))
    test_metrics = OrderedDict(((k, dict(out["test"][k])) for k in test_order))
    return train_metrics, test_metrics
