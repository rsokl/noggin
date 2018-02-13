from numbers import Integral, Real
import time
import importlib
import numpy as np
from inspect import cleandoc
from collections import OrderedDict, namedtuple
import os


class LiveMetric(object):
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
        value = np.asscalar(value)
        self._batch_data.append(value)
        self._running_weighted_sum += weighting*value
        self._total_weighting += weighting

    def set_epoch_datapoint(self, x=None):
        """ Parameters
            ----------
            x : Optional[Real]
                Specify the domain-value to be set for this data point."""
        self._epoch_data.append(self._running_weighted_sum / self._total_weighting)
        self._epoch_domain.append(x if x is not None else self.batch_domain[-1])
        self._running_weighted_sum = 0.
        self._total_weighting = 0.


class LivePlot:
    """ Plots batch-level and epoch-level summary statistics of the training and
        testing metrics of a model during a session.

        This class instance can be referenced after the training session to access the training
        performance statistics.

        Attributes
        ----------
        train_metrics : OrderedDict[str, Dict[str, numpy.ndarray]]
            Stores training metric results data and plot-objects.

        test_metrics : OrderedDict[str, Dict[str, numpy.ndarray]]
            Stores testing metric results data and plot-objects.

        pyplot : module
            Submodule of matplotlib

        Notes
        -----
        Live plotting is only supported for the 'nbAgg' backend (i.e.
        when the cell magic ``%matplotlib notebook`` is invoked in a
        jupyter notebook). When using other backends, a single plot will
        be produced at the end of the session. It is strongly recommended
        that only the 'nbAgg' backend is used - other backends may produce
        highly unstable behavior. """

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

    def plot_objects(self):
        """ The figure-instance of the plot, and the axis-instance for each metric.

            Returns
            -------
            Tuple[matplotlib.figure.Figure, numpy.ndarray(matplotlib.axes.Axes)]"""
        return self._fig, np.array(tuple(self._axis_mapping.values()))

    def __init__(self, metrics, refresh=0., plot_title=None, figsize=None, output_file=None,
                 each_batch_save=None):
        """ Parameters
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
                """

        # import matplotlib and check backend
        self.pyplot = importlib.import_module('matplotlib.pyplot')
        _matplotlib = importlib.import_module('matplotlib')

        self._backend = _matplotlib.get_backend()
        if 'nbAgg' not in self._backend:
            _inline_msg = """ Warning: live plotting is not supported when matplotlib uses the '{}'
                             backend. Instead, use the 'nbAgg' backend.
            
                             In a Jupyter notebook, this can be activated using the cell magic:
            
                                %matplotlib notebook."""
            print(cleandoc(_inline_msg.format(self._backend)))

        # type checking on inputs
        metrics = (metrics,) if isinstance(metrics, str) else tuple(metrics)
        assert all(isinstance(i, str) for i in metrics)
        assert isinstance(refresh, Real)
        assert plot_title is None or isinstance(plot_title, str)
        assert figsize is None or len(figsize) == 2 and all(isinstance(i, Integral) for i in figsize)
        assert output_file is None or isinstance(output_file, str)
        assert each_batch_save is None or isinstance.items()(each_batch_save, Integral) and each_batch_save > 0

        # input parameters
        self._metrics = metrics
        self._refresh = refresh
        self._liveplot = self._refresh >= 0. and 'nbAgg' in self._backend
        self._pltkwargs = {"figsize": figsize}
        self._plot_title = plot_title
        self._save_rate = each_batch_save

        # plot config
        self._batch_ax = dict(ls='-', alpha=0.5)  # plot settings for batch-data
        self._epoch_ax = dict(ls='-', marker='o', markersize=6, lw=3)  # plot settings for epoch-data
        self._legend = dict()

        # attribute initialization
        self._start_time = None  # Time upon entering the training session
        self._last_plot_time = None  # Time of last plot

        self._train_epoch_num = 0  # Current number of epochs trained
        self._train_batch_num = 0  # Current number of batches trained
        self._test_epoch_num = 0  # Current number of epochs tested
        self._test_batch_num = 0  # Current number of batches tested

        self._last_batch_acc = None  # Stores the previous batch accuracy
        self._last_epoch_acc = None  # Stores the previous epoch accuracy
        self._last_val_acc = None  # Stores the previous validation accuracy

        self._axis_mapping = OrderedDict()  # metric name -> matplotlib axis object

        # stores batch/epoch-level training statistics and plot objects for training/validation
        self._train_metrics = OrderedDict()
        self._test_metrics = OrderedDict()

        self._plot_batch = True

        self._fig, self._axes, self._text = None, None, None  # matplotlib plot objects

        ## TODO: implement pathlib.Path
        # prepping figure saving: self._file -> NamedTuple("File", [("pth", str), ("name", str), ("fmt", str)])
        if output_file is not None:
            File = namedtuple("File", ["pth", "name", "fmt"])
            pth, f = os.path.split(output_file)
            if not pth:
                pth = "."
            pth = os.path.abspath(pth)

            if not os.path.exists(pth):
                os.makedirs(pth)

            if not os.access(pth, os.W_OK):
                raise IOError("Permission denied: LivePlot does not have write access to {}".format(pth))

            try:
                name, fmt = f.rsplit(".")
                fmt = "." + fmt
            except ValueError:
                name, fmt = f, ""
            self._file = File(pth, name, fmt)
        else:
            self._file = None
            if self._save_rate is not None:
                raise ValueError("`each_batch_save` was specified, however `output_file` was not provided.")

    def __enter__(self):
        self._start_time = time.time()

        # enable active plotting
        if self.pyplot is not None:
            if self._liveplot:
                self.pyplot.ion()
            self._last_plot_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fig is not None:
            self._plot()
            if not self._liveplot:
                self.pyplot.show()
            self._save_fig()

    def set_train_batch(self, metrics, batch_size, plot=True):
        """
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
            """
        self._plot_batch = plot

        if not self._train_batch_num:
            self._init_plot_window()

            # initialize batch-level plot objects
            self._train_metrics.update((key, LiveMetric(key)) for key in metrics if key in self._metrics)
            for key, metric in self._train_metrics.items():
                try:
                    ax = self._axis_mapping[key]
                    metric.batch_line, = ax.plot([], [], label="train", **self._batch_ax)
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
            if self._save_rate and self._train_batch_num and self._train_batch_num % self._save_rate == 0:
                self._save_fig()

        self._train_batch_num += 1

    def plot_train_epoch(self):
        if not self._train_epoch_num:
            # initialize batch-level plot objects
            for ax, key in zip(self._axes, self._train_metrics):
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
                    metric.epoch_line, = ax.plot([], [], label="test", **self._epoch_ax)
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
        if self.pyplot is None or self._fig is not None:
            return None

        if self._start_time is None:
            self._start_time = time.time()

        self._fig, self._axes = self.pyplot.subplots(nrows=len(self._metrics), sharex=True, **self._pltkwargs)

        if len(self._metrics) == 1:
            self._axes = [self._axes]

        self._axis_mapping.update(zip(self._metrics, self._axes))

        for ax in self._axes:
            ax.grid(True)

        if self._plot_title:
            self._axes[0].set_title(self._plot_title)

        self._axes[-1].set_xlabel("Number of iterations")

        time_passed = time.strftime("%H:%M:%S", time.gmtime(0))
        text = "total time: {}\n".format(time_passed)
        self._text = self._axes[0].text(.3, .8, text,
                                        transform=self._axes[0].transAxes,
                                        bbox=dict(facecolor='none',
                                                  edgecolor='black',
                                                  boxstyle='round,pad=0.5'),
                                        family='monospace')

    def _resize(self):
        for ax in self._axes:
            ax.relim()
            ax.autoscale_view()

    def _update_text(self):
        time_passed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self._start_time))
        text = "total time: {}\n".format(time_passed)
        self._text.set_text(cleandoc(text))

    def _plot(self):
        if self.pyplot is None:
            return None

        # plot update all train/test line objects with latest x/y data
        for mode_metrics in [self._train_metrics, self._test_metrics]:
            for key, livedata in mode_metrics.items():
                if self._plot_batch and livedata.batch_line is not None:
                    livedata.batch_line.set_xdata(livedata.batch_domain)
                    livedata.batch_line.set_ydata(livedata.batch_data)

                if livedata.epoch_line is not None:
                    livedata.epoch_line.set_xdata(livedata.epoch_domain)
                    livedata.epoch_line.set_ydata(livedata.epoch_data)

        self._update_text()
        self._resize()
        if self._liveplot:
            self._fig.canvas.draw()

        self._last_plot_time = time.time()

    def _do_liveplot(self):
        # enable active plotting upon first plot
        if self._last_plot_time is None:
            if self._liveplot:
                self.pyplot.ion()
            self._last_plot_time = time.time()

        if self._liveplot and time.time() - self._last_plot_time >= self._refresh:
            self._plot()

    def _save_fig(self):
        if self._file is not None and self._fig is not None:
            # save figure
            _file = self._file.name + "_iter{}".format(self._train_batch_num) + self._file.fmt
            _file = os.path.join(self._file.pth, _file)
            self._fig.savefig(_file, bbox_inches="tight")
