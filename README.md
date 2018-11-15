[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rsokl/liveplot/master?filepath=LivePlot_Demo.ipynb)

# liveplot
Log and plot metrics during train/test time for a neural network (or whatever, really). `liveplot`
provides convenient i/o functions for saving, loading, and recreating liveplot sessions. It also provides
an interface for accessing logged metrics as [xarray data sets](http://xarray.pydata.org/en/stable/index.html). This
functionality, availabile via `liveplot.xarray`, permits users to seamlessly access their logged metrics as N-dimensional arrays with named axes.

![liveplot in action](images/demo.gif)

Please consult the [demo notebook](https://github.com/rsokl/LivePlot/blob/master/LivePlot_Demo.ipynb) for a summary of `liveplot`'s functionality.

## Installing liveplot
Clone/download this repository, navigate to the `liveplot` directory, and run:
```shell
python setup.py install
```

## Examples
### Creating a live plot
```python
import numpy as np
from liveplot import create_plot

%matplotlib notebook

metrics = {"accuracy":dict(train="C2", test="C3"),
           "loss": None}

plotter, fig, ax = create_plot(metrics)

for i, x in enumerate(np.linspace(0, 10, 100)):
    # training
    x += np.random.rand(1)*5
    batch_metrics = {"accuracy": x**2, "loss": 1/x**.5}
    plotter.set_train_batch(batch_metrics, batch_size=1, plot=True)

    # cue training epoch
    if i%10 == 0 and i > 0:
        plotter.plot_train_epoch()

        # cue test-time computations
        for x in np.linspace(0, 10, 5):
            x += (np.random.rand(1) - 0.5)*5
            test_metrics = {"accuracy": x**2}
            plotter.set_test_batch(test_metrics, batch_size=1)
        plotter.plot_test_epoch()

plotter.plot()  # ensures final data gets plotted


from liveplot import save_metrics

# save metrics from liveplot instance
save_metrics("tmp.npz", plotter)
```

### Recreating plot from saved metrics
```python
from liveplot import load_metrics, recreate_plot

train, test = load_metrics("tmp.npz")

colors = {"accuracy":dict(train="C4", test="C6"),
          "loss":"red"}

new_plotter, fig, ax = recreate_plot(train_metrics=train,
                                     test_metrics=test,
                                     colors=colors)
```

## Documentation

```
def create_plot(metrics, refresh=0., plot_title=None, figsize=None, track_time=True):
    """ Create matplotlib figure/axes, and a live-plotter, which publishes
        "live" training/testing metric data, at a batch and epoch level, to
        the figure.

        Parameters
        ----------
        metrics : Union[str, Sequence[str], Dict[str, valid-color]
            The name, or sequence of names, of the metric(s) that will be plotted.

            `metrics` can also be a dictionary, specifying the colors used to plot
            the metrics. Two mappings are valid:
                - metric-name -> color-value  (specifies train-metric color only)
                - metric-name -> {train/test : color-value}

        refresh : float, optional (default=0.)
            Sets the plot refresh rate in seconds.

            A refresh rate of 0. updates the once every 1/1000 seconds. A
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


def recreate_plot(liveplot=None, *, train_metrics=None, test_metrics=None, colors=None):
    """ Recreate a plot from a LivePlot instance or from train/test metric dictionaries.

        Parameters
        ----------
        liveplot : Optional[liveplot.LivePlot]
            An existing liveplot object.

        Keyword-Only Arguments
        ----------------------
        train_metrics : Optional[OrderedDict[str, Dict[str, numpy.ndarray]]]
           metric_name -> {"batch_data":   array,
                           "epoch_data":   array,
                           "epoch_domain": array}

        test_metrics : Optional[OrderedDict[str, Dict[str, numpy.ndarray]]]
           metric_name -> {"batch_data":   array,
                           "epoch_data":   array,
                           "epoch_domain": array}

        colors : Optional[Dict[str, color-value], Dict[str, Dict[str, color-value]]
            Specifying train-time metric colors only:
                  metric-name -> color-value

            Specifying train or test-time metric colors:
                 metric-name -> {train/test -> color-value}

        Returns
        -------
        Tuple[liveplot.LivePlot, matplotlib.figure.Figure, numpy.ndarray(matplotlib.axes.Axes)]
            (LivePlot-instance, figure, array-of-axes)"""



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

            metric-name -> {batch_data -> array, epoch_domain -> array, epoch_data -> array}

        test_metrics : Optional[OrderedDict[str, Dict[str, numpy.ndarray]]]]
            metric-name -> {batch_data -> array, epoch_domain -> array, epoch_data -> array}"""


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

class LivePlot:
    """ Plots batch-level and epoch-level summary statistics of the training and
        testing metrics of a model during a session.


        Attributes
        ----------
        train_metrics : OrderedDict[str, Dict[str, numpy.ndarray]]
            Stores training metric results data and plot-objects.

        test_metrics : OrderedDict[str, Dict[str, numpy.ndarray]]
            Stores testing metric results data and plot-objects.

        metric_colors : Dict[str, Dict[str, color-value]]
            {metric-name -> {'train'/'test' -> color-value}}

       Methods
       -------
       plot(self)
           Plot data, irrespective of the refresh rate. This should only
           be called if you are generating a static plot.

       plot_objects(self)
           The figure-instance of the plot, and the axis-instance for each metric.

           Returns
           -------
           Tuple[matplotlib.figure.Figure, numpy.ndarray(matplotlib.axes.Axes)]

       plot_test_epoch(self)

       plot_train_epoch(self)

       set_test_batch(self, metrics, batch_size)

           Parameters
           ----------
           metrics : Dict[str, Real]
               Mapping of metric-name to value. Only those metrics that were
               registered when initializing LivePlot will be recorded.

           batch_size : Integral
               The number of samples in the batch used to produce the metrics.
               Used to weight the metrics to produce epoch-level statistics.

       set_train_batch(self, metrics, batch_size, plot=True)

           Parameters
           ----------
           metrics : Dict[str, Real]
               Mapping of metric-name to value. Only those metrics that were
               registered when initializing LivePlot will be recorded.

           batch_size : Integral
               The number of samples in the batch used to produce the metrics.
               Used to weight the metrics to produce epoch-level statistics.

           plot : bool
               If True, plot the batch-metrics (adhering to the refresh rate)

       show(self)
           Calls `matplotlib.pyplot.show()`. For visualizing a static-plot. """```
