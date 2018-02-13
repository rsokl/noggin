from liveplot.plotter import LivePlot


def create_plot(metrics, refresh=0., plot_title=None, figsize=None):
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

        Returns
        -------
        Tuple[liveplot.LivePlot, matplotlib.figure.Figure, numpy.ndarray(matplotlib.axes.Axes)]
            (LivePlot-instance, figure, array-of-axes) """

    plot = LivePlot(metrics, refresh, plot_title, figsize)
    plot._init_plot_window()
    fig, ax = plot.plot_objects()
    return plot, fig, ax