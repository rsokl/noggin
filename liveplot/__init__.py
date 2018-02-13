from liveplot.plotter import LivePlot

def create_plot(metrics, refresh=0.,
                plot_title=None,
                figsize=None,
                output_file=None,
                each_batch_save=None):
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

        output_file : Optional[str], optional (default="fig.png")
            Required for saving the figure. Specifies the path, base filename,
            and output format of the saved figure.

            The iteration number is appended to the end of the base filename
            (e.g. 'path/fig_100.png')

        each_batch_save : Optional[int]
            The iteration-rate at which the plot figure is saved.

            If None is specified but `output_file` is specified, the figure
            will be saved at only the end of the session.

        Returns
        -------
        Tuple[liveplot.LivePlot, matplotlib.figure.Figure, numpy.ndarray(matplotlib.axes.Axes)]
            (LivePlot-instance, figure, array-of-axes) """
    plotter = LivePlot(metrics, refresh, plot_title, figsize, output_file, each_batch_save)
    plotter._init_plot_window()
    fig, ax = plotter.plot_objects()
    return plotter, fig, ax