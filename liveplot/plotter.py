from numbers import Integral, Real
import time
import importlib
import numpy as np
from inspect import cleandoc
from collections import OrderedDict, defaultdict
import warnings
from liveplot.utils import check_valid_color

class LiveMetric:
    """ Holds the relevant data for a train/test metric for live plotting. """
    def __init__(self, name):
        """ Parameters
            ----------
            name : str """
        self.name = name
        self.batch_line = None   # ax object for batch data
        self.epoch_line = None   # ax object for epoch data
        self._batch_data = []   # metric data for consecutive batches
        self._epoch_data = []   # accuracies
        self._epoch_domain = []
        self._running_weighted_sum = 0.
        self._total_weighting = 0.

    @property
    def batch_domain(self):
        return np.arange(1, len(self.batch_data) + 1, dtype=float)

    @property
    def batch_data(self):
        """ Metric data for consecutive batches.

            Returns
            -------
            numpy.ndarray, shape=(N_batch, )"""
        return np.array(self._batch_data)

    @property
    def epoch_domain(self):
        return np.array(self._epoch_domain)

    @property
    def epoch_data(self):
        """ Metric data for consecutive epochs.

            Returns
            -------
            numpy.ndarray, shape=(N_epoch, )"""
        return np.array(self._epoch_data)

    def add_datapoint(self, value, weighting=1.):
        """ Parameters
            ----------
            value : Real
            weighting : Real """
        if isinstance(value, np.ndarray):
            value = np.asscalar(value)

        self._batch_data.append(value)
        self._running_weighted_sum += weighting*value
        self._total_weighting += weighting

    def set_epoch_datapoint(self, x=None):
        """ Parameters
            ----------
            x : Optional[Real]
                Specify the domain-value to be set for this data point."""
        if self._running_weighted_sum:
            self._epoch_data.append(self._running_weighted_sum / self._total_weighting)
            self._epoch_domain.append(x if x is not None else self.batch_domain[-1])
            self._running_weighted_sum = 0.
            self._total_weighting = 0.


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

        Notes
        -----
        Live plotting is only supported for the 'nbAgg' backend (i.e.
        when the cell magic ``%matplotlib notebook`` is invoked in a
        jupyter notebook). """

    @property
    def train_metrics(self):
        """ The batch and epoch data for each metric.

            Returns
            -------
            OrderedDict[str, Dict[str, numpy.ndarray]]

            metric-name -> {batch_data -> array, epoch_domain -> array, epoch_data -> array} """
        out = OrderedDict()
        for k, v in self._train_metrics.items():
            out[k] = {attr: getattr(v, attr) for attr in ["batch_data", "epoch_data", "epoch_domain"]}
        return out

    @property
    def test_metrics(self):
        """ The batch and epoch data for each metric.

            Returns
            -------
            OrderedDict[str, Dict[str, numpy.ndarray]]

            metric-name -> {batch_data -> array, epoch_domain -> array, epoch_data -> array} """
        out = OrderedDict()
        for k, v in self._test_metrics.items():
            out[k] = {attr: getattr(v, attr) for attr in ["batch_data", "epoch_data", "epoch_domain"]}
        return out

    @property
    def metric_colors(self):
        """ Returns
            -------
            Dict[str, Dict[str, color-value]]
                {metric-name -> {'train'/'test' -> color-value}}"""
        out = defaultdict(dict)
        for k, v in self._train_colors.items():
            out[k]["train"] = v
        
        for k, v in self._test_colors.items():
                    out[k]["test"] = v
        return dict(out)


    @property
    def refresh(self):
        return self._refresh

    @refresh.setter
    def refresh(self, value):
        """ Set the refresh rate (per second). A negative refresh rate
            turns off static plotting."""
        assert isinstance(value, Real)
        self._refresh = value
        self._liveplot = self._refresh >= 0. and 'nbAgg' in self._backend

    def plot_objects(self):
        """ The figure-instance of the plot, and the axis-instance for each metric.

            Returns
            -------
            Tuple[matplotlib.figure.Figure, numpy.ndarray(matplotlib.axes.Axes)]"""
        return self._fig, np.array(tuple(self._axis_mapping.values()))

    def __init__(self, metrics, refresh=0., plot_title=None, figsize=None, track_time=True):
        """ Parameters
            ----------
            metrics : Union[str, Sequence[str], Dict[str, valid-color]
                The name, or sequence of names, of the metric(s) that will be plotted.

                `metrics` can also specify the colors used to plot the metrics via the mappings:
                    - metric-name -> color-value  (specifies train-metric color only)
                    - metric-name -> {train/test -> color-value}

            refresh : float, optional (default=0.)
                Sets the plot refresh rate in seconds.

                A refresh rate of 0. updates the once every 1/1000 seconds.

                A negative refresh rate  turns off live plotting:
                   Call `self.plot()` to draw the static plot.
                   Call `self.show()` to open a window showing the static plot

            plot_title : Optional[str]
                Specifies the title used on the plot.

            figsize : Optional[Sequence[int, int]]
                Specifies the width and height, respectively, of the figure.

            track_time : bool, default=True
                If `True`, the total time of plotting is annotated in within the first axes"""

        # type checking on inputs

        assert isinstance(refresh, Real)
        assert plot_title is None or isinstance(plot_title, str)
        assert figsize is None or len(figsize) == 2 and all(isinstance(i, Integral) for i in figsize)
        assert isinstance(track_time, bool)

        # import matplotlib and check backend
        self._pyplot = importlib.import_module('matplotlib.pyplot')
        _matplotlib = importlib.import_module('matplotlib')

        self._backend = _matplotlib.get_backend()
        if 'nbAgg' not in self._backend and refresh >= 0:
            _inline_msg = """ Warning: live plotting is not supported when matplotlib uses the '{}'
                             backend. Instead, use the 'nbAgg' backend.
            
                             In a Jupyter notebook, this can be activated using the cell magic:
            
                                %matplotlib notebook."""
            print(cleandoc(_inline_msg.format(self._backend)))

        # input parameters
        self._metrics = (metrics,) if isinstance(metrics, str) else tuple(metrics)
        if any(not isinstance(i, str) for i in self._metrics):
            raise TypeError("`metrics` must be a string or a collection of strings")

        if 0 <= refresh < 0.001:
            refresh = 0.001
        self._refresh = refresh
        self._liveplot = self._refresh >= 0. and 'nbAgg' in self._backend
        self._pltkwargs = {"figsize": figsize}
        self._plot_title = plot_title
        self._track_time = track_time

        # plot config
        self._train_colors = defaultdict(lambda: None)
        self._test_colors = defaultdict(lambda: None)
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, dict):
                    self._train_colors[k] = v.get("train")
                    self._test_colors[k] = v.get("test")
                else:
                    self._train_colors[k] = v
        sum(check_valid_color(c) for c in self._train_colors.values())
        sum(check_valid_color(c) for c in self._test_colors.values())

        self._batch_ax = dict(ls='-', alpha=0.5)  # plot settings for batch-data
        self._epoch_ax = dict(ls='-', marker='o', markersize=6, lw=3)  # plot settings for epoch-data
        self._legend = dict()
        self._axis_mapping = OrderedDict()  # metric name -> matplotlib axis object
        self._plot_batch = True
        self._fig, self._axes, self._text = None, None, None  # matplotlib plot objects

        # attribute initialization
        self._start_time = None      # float: Time upon entering the training session
        self._last_plot_time = None  # float: Time of last plot

        self._train_epoch_num = 0    # int: Current number of epochs trained
        self._train_batch_num = 0    # int: Current number of batches trained
        self._test_epoch_num = 0     # int: Current number of epochs tested
        self._test_batch_num = 0     # int: Current number of batches tested

        # stores batch/epoch-level training statistics and plot objects for training/testing
        self._train_metrics = OrderedDict()  # metric-name -> LiveMetric
        self._test_metrics = OrderedDict()   # metric-name -> LiveMetric

    def __repr__(self):
        msg = "LivePlot({})\n\n".format(", ".join(self._metrics))

        words = ["training batches", "training epochs", "testing batches", "testing epochs"]
        things = [self._train_batch_num, self._train_epoch_num,
                  self._test_batch_num, self._test_epoch_num]

        for word, thing in zip(words, things):
            msg += "number of {word} set: {thing}\n".format(word=word, thing=thing)

        if self._track_time and self._last_plot_time is not None:
            t = time.strftime("%H:%M:%S", time.localtime(self._last_plot_time))
            msg += "\n\nlast plot time: {}\n".format(t)
        return msg

    def set_train_batch(self, metrics, batch_size, plot=True):
        """ Parameters
            ----------
            metrics : Dict[str, Real]
                Mapping of metric-name to value. Only those metrics that were
                registered when initializing LivePlot will be recorded.

            batch_size : Integral
                The number of samples in the batch used to produce the metrics.
                Used to weight the metrics to produce epoch-level statistics.

            plot : bool
                If True, plot the batch-metrics (adhering to the refresh rate)"""
        self._plot_batch = plot

        if not self._train_batch_num:
            self._init_plot_window()

            unreg_metrics = set(metrics).difference(self._metrics)
            if unreg_metrics:
                msg = "\nThe following training metrics are not registered for live-plotting:\n\t" + "\n\t".join(sorted(unreg_metrics))
                warnings.warn(cleandoc(msg),)

            # initialize batch-level plot objects
            self._train_metrics.update((key, LiveMetric(key)) for key in metrics if key in self._metrics)
            for key, metric in self._train_metrics.items():
                try:
                    ax = self._axis_mapping[key]
                    metric.batch_line, = ax.plot([], [], label="train", color=self._train_colors.get(key), **self._batch_ax)
                    ax.set_ylabel(key)
                    ax.legend()
                except KeyError:
                    pass

        # record each incoming batch metric
        for key, value in metrics.items():
            try:
                self._train_metrics[key].add_datapoint(value, weighting=batch_size)
            except KeyError:
                pass

        if self._plot_batch:
            self._do_liveplot()

        self._train_batch_num += 1

    def plot_train_epoch(self):
        if not self._train_epoch_num:
            # initialize batch-level plot objects
            for key in self._train_metrics:
                ax = self._axis_mapping[key]
                batch_color = self._train_metrics[key].batch_line.get_color()
                self._train_metrics[key].epoch_line, = ax.plot([], [], color=batch_color, **self._epoch_ax)
                ax.legend(**self._legend)

        # compute epoch-mean metrics
        for key in self._train_metrics:
            self._train_metrics[key].set_epoch_datapoint()

        self._do_liveplot()
        self._train_epoch_num += 1

    def set_test_batch(self, metrics, batch_size):
        """
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
        if not self._test_batch_num:
            self._test_metrics.update((key, LiveMetric(key)) for key in metrics if key in self._metrics)

            unreg_metrics = set(metrics).difference(self._metrics)
            if unreg_metrics:
                msg = "\nThe following testing metrics are not registered for live-plotting:\n\t" + "\n\t".join(sorted(unreg_metrics))
                warnings.warn(cleandoc(msg),)

        # record each incoming batch metric
        for key, value in metrics.items():
            try:
                self._test_metrics[key].add_datapoint(value, weighting=batch_size)
            except KeyError:
                pass

        self._test_batch_num += 1

    def plot_test_epoch(self):
        if not self._test_epoch_num:
            self._init_plot_window()

            # initialize epoch-level plot objects
            for key, metric in self._test_metrics.items():
                try:
                    ax = self._axis_mapping[key]
                    metric.epoch_line, = ax.plot([], [], label="test", color=self._test_colors.get(key), **self._epoch_ax)
                    ax.set_ylabel(key)
                    ax.legend(**self._legend)
                except KeyError:
                    pass

        # compute epoch-mean metrics
        for key in self._test_metrics:
            try:
                x = self._train_metrics[key].batch_domain[-1] if self._train_metrics else None
            except KeyError:
                x = None
            self._test_metrics[key].set_epoch_datapoint(x)

        self._do_liveplot()
        self._test_epoch_num += 1

    def _init_plot_window(self):
        if self._pyplot is None or self._fig is not None:
            return None

        if self._start_time is None:
            self._start_time = time.time()

        self._fig, self._axes = self._pyplot.subplots(nrows=len(self._metrics), sharex=True, **self._pltkwargs)

        if len(self._metrics) == 1:
            self._axes = [self._axes]

        self._axis_mapping.update(zip(self._metrics, self._axes))

        for ax in self._axes:
            ax.grid(True)

        if self._plot_title:
            self._axes[0].set_title(self._plot_title)

        self._axes[-1].set_xlabel("Number of iterations")

        time_passed = time.strftime("%H:%M:%S", time.gmtime(0))

        if self._track_time:
            text = "total time: {}\n".format(time_passed)
            self._text = self._axes[0].text(.3, .8, text,
                                            transform=self._axes[0].transAxes,
                                            bbox=dict(facecolor='none',
                                                      edgecolor='none',
                                                      boxstyle='round,pad=0.5'),
                                            family='monospace')

    def _resize(self):
        for ax in self._axes:
            ax.relim()
            ax.autoscale_view()

    def _update_text(self):

        for ax in self._axis_mapping.values():
            ax.legend()

        if not self._track_time:
            return None

        time_passed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self._start_time))
        text = "total time: {}\n".format(time_passed)
        self._text.set_text(cleandoc(text))

    def plot(self):
        """ Plot data, irrespective of the refresh rate. This should only
           be called if you are generating a static plot."""
        if self._pyplot is None:
            return None

        # plot update all train/test line objects with latest x/y data
        for i, mode_metrics in enumerate([self._train_metrics, self._test_metrics]):
            for key, livedata in mode_metrics.items():

                if self._plot_batch and livedata.batch_line is not None:
                    livedata.batch_line.set_xdata(livedata.batch_domain)
                    livedata.batch_line.set_ydata(livedata._batch_data)
                    if livedata._epoch_data:
                        livedata.batch_line.set_label("train: {:.2e}".format(livedata._epoch_data[-1]))

                if livedata.epoch_line is not None:
                    livedata.epoch_line.set_xdata(livedata.epoch_domain)
                    livedata.epoch_line.set_ydata(livedata._epoch_data)
                    if i == 1 and livedata._epoch_data:
                        livedata.epoch_line.set_label("test: " + "{:.2e}".format(livedata._epoch_data[-1]))

        self._update_text()
        self._resize()
        if self._liveplot:
            self._fig.canvas.draw()

        self._last_plot_time = time.time()

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
