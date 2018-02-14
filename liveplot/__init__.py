from liveplot.plotter import LivePlot

__all__ = ["create_plot", "recreate_plot"]

def create_plot(metrics, refresh=0., plot_title=None, figsize=None, track_time=True):
    """ Create matplotlib figure/axes, and a live-plotter, which publishes
        "live" training/testing metric data, at a batch and epoch level, to
        the figure.

        Parameters
        ----------
        metrics : Union[str, Sequence[str]]
            The name, or sequence of names, of the metric(s) that will be plotted.

        refresh : float, optional (default=0.)
            Sets the plot refresh rate in seconds.

            A refresh rate of 0. updates the plot as frequently as possible. A
            negative refresh rate will draw the plot only at the end of the session.

        plot_title : Optional[str]
            Specifies the title used on the plot.

        figsize : Optional[Sequence[int, int]]
            Specifies the width and height, respectively, of the figure.

        track_time : bool, default=True
            If `True`, the total time of plotting is annotated in within the first axes

        Returns
        -------
        Tuple[liveplot.LivePlot, matplotlib.figure.Figure, numpy.ndarray(matplotlib.axes.Axes)]
            (LivePlot-instance, figure, array-of-axes)


        Examples
        --------
        >>> %matplotlib notebook
        >>> import numpy as np
        >>> from liveplot import create_plot
        >>> metrics = ["accuracy", "loss"]
        >>> plotter, fig, ax = create_plot(metrics, refresh=0)
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

        """

    live_plotter = LivePlot(metrics, refresh, plot_title, figsize, track_time)
    live_plotter._init_plot_window()
    fig, ax = live_plotter.plot_objects()
    return live_plotter, fig, ax


def recreate_plot(old_plot=None, train_metrics=None, test_metrics=None):
    """ Recreate a plot from a LivePlot instance or from train/test metric dictionaries.

        Parameters
        ----------
        old_plot : Optional[liveplot.LivePlot]
            An existing liveplot object.

        train_metrics : Optional[OrderedDict[str, Dict[str, numpy.ndarray]]]
           metric_name -> {"batch_data":   array,
                           "epoch_data":   array,
                           "epoch_domain": array}

        test_metrics : Optional[OrderedDict[str, Dict[str, numpy.ndarray]]]
           metric_name -> {"batch_data":   array,
                           "epoch_data":   array,
                           "epoch_domain": array}

        Returns
        -------
        Tuple[liveplot.LivePlot, matplotlib.figure.Figure, numpy.ndarray(matplotlib.axes.Axes)]
            (LivePlot-instance, figure, array-of-axes)"""
    from liveplot.plotter import LiveMetric
    assert any(i is not None for i in [old_plot, train_metrics, test_metrics])
    if old_plot is not None:
        assert train_metrics is None and test_metrics is None
    else:
        if train_metrics is None:
            train_metrics = {}
        if test_metrics is None:
            test_metrics = {}

    if old_plot:
        metrics = old_plot._metrics
    else:
        metrics = list(train_metrics)
        metrics.extend(k for k in test_metrics if k not in metrics)

    new, fig, ax = create_plot(metrics, refresh=-1)

    if old_plot:
        new._metric_colors = old_plot._metric_colors

    new._init_plot_window()
    fig, ax = new.plot_objects()

    if old_plot:
        new._train_metrics = old_plot._train_metrics
        new._test_metrics = old_plot._test_metrics

        for attr in ["_train_epoch_num", "_train_batch_num",
                     "_test_epoch_num", "_test_batch_num"]:
            setattr(new, attr, getattr(old_plot, attr))
    else:
        new._train_metrics.update((key, LiveMetric(key)) for key in train_metrics if key in new._metrics)
        new._test_metrics.update((key, LiveMetric(key)) for key in test_metrics if key in new._metrics)

        for metric_name, metric in new._train_metrics.items():
            for item in ["batch_data", "epoch_data", "epoch_domain"]:
                setattr(metric, "_" + item, list(train_metrics[metric_name][item]))

        for metric_name, metric in new._test_metrics.items():
            for item in ["batch_data", "epoch_data", "epoch_domain"]:
                setattr(metric, "_" + item, list(test_metrics[metric_name][item]))

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

    for key, metric in new._train_metrics.items():
        try:
            ax = new._axis_mapping[key]
            metric.batch_line, = ax.plot([], [], label="train", color=new._metric_colors[key], **new._batch_ax)
            ax.set_ylabel(key)
            ax.legend()
        except KeyError:
            pass

    for key in new._train_metrics:
        ax = new._axis_mapping[key]
        batch_color = new._train_metrics[key].batch_line.get_color()
        new._train_metrics[key].epoch_line, = ax.plot([], [], color=batch_color, **new._epoch_ax)
        ax.legend(**new._legend)

    for key, metric in new._test_metrics.items():
        try:
            ax = new._axis_mapping[key]
            metric.epoch_line, = ax.plot([], [], label="test", **new._epoch_ax)
            ax.set_ylabel(key)
            ax.legend(**new._legend)
        except KeyError:
            pass

    new.plot()
    return new, fig, ax