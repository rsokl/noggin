from liveplot.plotter import LivePlot


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