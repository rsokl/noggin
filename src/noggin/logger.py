"""
Provides functionality for logging training and testing batch-level & epoch-level metrics
"""

from collections import OrderedDict
from itertools import product
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy import ndarray

try:
    from xarray import Dataset
except ImportError:  # pragma: no cover
    Dataset = Any

__all__ = ["LiveLogger", "LiveMetric"]


class LiveMetric:
    """ Holds the relevant data for a train/test metric for live plotting. """

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str

        Raises
        ------
        TypeError
            Invalid metric name (must be string)
            """
        if not isinstance(name, str):
            raise TypeError(
                "Metric names must be specified as strings. Got: {}".format(name)
            )
        self._name = name

        # axis objects for batch/epoch data
        self.batch_line = None
        self.epoch_line = None

        self._batch_data_list = []  # type: List[float]
        self._batch_data = np.array([])  # type: np.ndarray

        self._epoch_data_list = []  # type: List[float]
        self._epoch_data = np.array([])  # type: np.ndarray

        self._epoch_domain_list = []  # type: List[int]
        self._epoch_domain = np.array([])  # type: np.ndarray

        self._running_weighted_sum = 0.0
        self._total_weighting = 0.0
        self._cnt_since_epoch = 0

    @property
    def name(self) -> str:
        """Name of the metric.

        Returns
        -------
        str"""
        return self._name

    @property
    def batch_domain(self) -> ndarray:
        """Array of iteration-counts at which the metric was recorded.

        Returns
        -------
        numpy.ndarray, shape=(N_batch, )
        """
        return np.arange(1, len(self.batch_data) + 1, dtype=float)

    @property
    def batch_data(self) -> ndarray:
        """Batch-level measurements of the metric.

        Returns
        -------
        numpy.ndarray, shape=(N_batch, )"""
        if self._batch_data_list:
            self._batch_data = np.concatenate((self._batch_data, self._batch_data_list))
            self._batch_data_list = []
        return self._batch_data

    @property
    def epoch_domain(self) -> ndarray:
        """Array of iteration-counts at which an epoch was set for this metric.

        Returns
        -------
        numpy.ndarray, shape=(N_epoch, )"""
        if self._epoch_domain_list:
            self._epoch_domain = np.concatenate(
                (self._epoch_domain, self._epoch_domain_list)
            )
            self._epoch_domain_list = []
        return self._epoch_domain

    @property
    def epoch_data(self) -> ndarray:
        """Epoch-level measurements of the metrics.

        When an epoch is set, the mean-value of the metric is computed over
        all of its measurements since the last recorded epoch.

        Returns
        -------
        numpy.ndarray, shape=(N_epoch, )"""
        if self._epoch_data_list:
            self._epoch_data = np.concatenate((self._epoch_data, self._epoch_data_list))
            self._epoch_data_list = []
        return self._epoch_data

    def add_datapoint(self, value: Real, weighting: Real = 1.0):
        """Record a batch-level measurement of the metric.

        Parameters
        ----------
        value : Real
            The recorded value.
        weighting : Real
            The weight with which this recorded value will contribute
            to the epoch-level mean."""
        if isinstance(value, np.ndarray):
            value = value.item()

        self._batch_data_list.append(value)
        self._running_weighted_sum += weighting * value
        self._total_weighting += weighting
        self._cnt_since_epoch += 1

    def set_epoch_datapoint(self, x: Optional[Real] = None):
        """Mark the present iteration as an epoch, and compute
        the mean value of the metric since the past epoch.

        Parameters
        ----------
        x : Optional[Real]
            Specify the domain-value to be set for this data point."""
        if self._cnt_since_epoch > 0:
            mean = self._running_weighted_sum / (
                self._total_weighting if self._total_weighting > 0.0 else 1.0
            )
            self._epoch_data_list.append(mean)
            self._epoch_domain_list.append(
                x if x is not None else self.batch_domain[-1]
            )
            self._running_weighted_sum = 0.0
            self._total_weighting = 0.0
            self._cnt_since_epoch = 0

    def to_dict(self) -> Dict[str, ndarray]:
        """ Returns the batch data, epoch domain, and epoch data
        in a dictionary.

        Additionally, running statistics are included in order to
        preserve the state of the metric.

        Returns
        -------
        Dict[str, ndarray]

        Notes
        -----
        The encoded dictionary stores::

            'batch_data' -> ndarray, shape-(N,)
            'epoch_data' -> ndarray, shape-(M,)
            'epoch_domain' -> ndarray, shape-(M,)
            'cnt_since_epoch' -> int
            'total_weighting' -> float
            'running_weighted_sum' -> float
            'name' -> str
        """
        out = {
            attr: getattr(self, attr)
            for attr in ("batch_data", "epoch_data", "epoch_domain")
        }
        out.update(
            (attr, getattr(self, "_" + attr))
            for attr in (
                "cnt_since_epoch",
                "total_weighting",
                "running_weighted_sum",
                "name",
            )
        )
        return out

    @classmethod
    def from_dict(cls, metrics_dict: Dict[str, ndarray]):
        """ The inverse of ``LiveMetric.to_dict``. Given a dictionary of
        live-metric data, constructs an instance of `LiveMetric`.

        Parameters
        ----------
        metrics_dict: Dict[str, ndarray]
            Stores the state of the live-metric instance being created.

        Returns
        -------
        noggin.LiveMetric

        Notes
        -----
        The encoded dictionary stores::

            'batch_data' -> ndarray, shape-(N,)
            'epoch_data' -> ndarray, shape-(M,)
            'epoch_domain' -> ndarray, shape-(M,)
            'cnt_since_epoch' -> int
            'total_weighting' -> float
            'running_weighted_sum' -> float
            'name' -> str
        """
        array_keys = ("batch_data", "epoch_data", "epoch_domain")
        running_stats_keys = (
            "running_weighted_sum",
            "total_weighting",
            "cnt_since_epoch",
        )
        required_keys = array_keys + running_stats_keys

        if not isinstance(metrics_dict, dict):
            raise TypeError(
                "`live_metrics` must be a dictionary, "
                "got type {}".format(type(metrics_dict))
            )

        if not (set(required_keys) <= set(metrics_dict)):
            raise ValueError(
                "`live_metrics` is missing the following keys: "
                "'{}'".format(", ".join(set(required_keys) - set(metrics_dict)))
            )

        out = cls(metrics_dict["name"])
        for k in required_keys:
            v = metrics_dict[k]
            if k in array_keys:
                if not isinstance(v, np.ndarray) or v.ndim != 1:
                    raise ValueError("'{}' must map to a 1D numpy arrays".format(k))
            else:
                if not isinstance(v, Real):
                    raise ValueError("'{}' must map to a real number".format(k))
                if k == "cnt_since_epoch" and (not isinstance(v, Integral) or v < 0):
                    raise ValueError("{} must map to a non-negative value".format(k))
            setattr(out, "_" + k, v)
        return out


class LiveLogger:
    """
    Logs batch-level and epoch-level summary statistics of the training and
    testing metrics of a model during a session.

    Examples
    --------
    A simple example in which we log two iterations of training batches,
    and set an epoch.

    >>> from noggin import LiveLogger
    >>> logger = LiveLogger()
    >>> logger.set_train_batch(dict(metric_a=2., metric_b=1.), batch_size=10)
    >>> logger.set_train_batch(dict(metric_a=0., metric_b=2.), batch_size=4)
    >>> logger.set_train_epoch()  # compute the mean statistics
    >>> logger
    LiveLogger(metric_a, metric_b)
    number of training batches set: 2
    number of training epochs set: 1
    number of testing batches set: 0
    number of testing epochs set: 0

    Accessing our logged batch-level and epoch-level data

    >>> logger.to_xarray("train")
    MetricArrays(batch=<xarray.Dataset>
    Dimensions:     (iterations: 2)
    Coordinates:
      * iterations  (iterations) int32 1 2
    Data variables:
        metric_a    (iterations) float64 2.0 0.0
        metric_b    (iterations) float64 1.0 2.0,
    epoch=<xarray.Dataset>
    Dimensions:     (iterations: 1)
    Coordinates:
      * iterations  (iterations) int32 2
    Data variables:
        metric_a    (iterations) float64 1.429
        metric_b    (iterations) float64 1.286)
    """

    @property
    def train_metrics(self) -> Dict[str, Dict[str, ndarray]]:
        """
        The batch and epoch data for each train-metric.

        Returns
        -------
        OrderedDict[str, Dict[str, numpy.ndarray]]
            The structure of the resulting dictionary is::

                '<metric-name>' -> {"batch_data":   array,
                                    "epoch_data":   array,
                                    "epoch_domain": array,
                                    ...} """
        return OrderedDict((k, v.to_dict()) for k, v in self._train_metrics.items())

    @property
    def test_metrics(self) -> Dict[str, Dict[str, ndarray]]:
        """
        The batch and epoch data for each test-metric.

        Returns
        -------
        OrderedDict[str, Dict[str, numpy.ndarray]]
            The structure of the resulting dictionary is::

                '<metric-name>' -> {"batch_data":   array,
                                    "epoch_data":   array,
                                    "epoch_domain": array,
                                    ...} """
        return OrderedDict((k, v.to_dict()) for k, v in self._test_metrics.items())

    def to_xarray(self, train_or_test: str) -> Tuple[Dataset, Dataset]:
        """
        Returns xarray datasets for the batch-level and epoch-level metrics, respectively,
        for either the train-metrics or test-metrics.

        Parameters
        ----------
        train_or_test : str
            Either 'train' or 'test' - specifies which measurements to be returned

        Returns
        -------
        Tuple[xarray.Dataset, xarray.Dataset]
            The batch-level and epoch-level datasets. The metrics are reported as
            data variables in the dataset, and the coordinates corresponds to
            the batch-iteration count.

        Notes
        -----
        The layout of the resulting data sets are::

            Dimensions:     (iterations: num_iterations)
            Coordinates:
              * iterations  (iterations) int64 1 2 3 ...
            Data variables:
                metric0      (iterations) float64 val_0 val_1 ...
                metric1      (iterations) float64 val_0 val_1 ...
                ...

        Each metric can be accessed as an attribute of the resulting data-set,
        e.g. ``dataset.metric0``, or via the 'get-item' syntax, e.g.
        ``dataset['metric0']``. This returns a data-array for that metric.

        Data sets collected from multiple trials of an experiment can be combined
        using :func:`~noggin.xarray.concat_experiments`.
        """
        from .xarray import metrics_to_xarrays

        if train_or_test not in ["train", "test"]:
            raise ValueError(
                "`train_or_test` must be 'train' or 'test',"
                "\nGot: {}".format(train_or_test)
            )
        metrics = self.train_metrics if train_or_test == "train" else self.test_metrics
        return metrics_to_xarrays(metrics)

    def to_dict(self) -> Dict[str, Any]:
        """Records the state of the logger in a dictionary.

        This is the inverse of :func:`~noggin.logger.LiveLogger.from_dict`

        Returns
        -------
        Dict[str, Any]

        Notes
        -----
        To save your logger, use this method to convert it to a dictionary
        and then pickle the dictionary.
        """
        return dict(
            train_metrics=self.train_metrics,
            test_metrics=self.test_metrics,
            num_train_epoch=self._num_train_epoch,
            num_train_batch=self._num_train_batch,
            num_test_batch=self._num_test_batch,
            num_test_epoch=self._num_test_epoch,
        )

    @classmethod
    def from_dict(cls, logger_dict: Dict[str, Any]):
        """Records the state of the logger in a dictionary.

        This is the inverse of :func:`~noggin.logger.LiveLogger.to_dict`

        Parameters
        ----------
        logger_dict : Dict[str, Any]
            The dictionary storing the state of the logger to be
            restored.

        Returns
        -------
        noggin.LiveLogger
            The restored logger.

        Notes
        -----
        This is a class-method, the syntax for invoking it is:

        >>> LiveLogger.from_dict(logger_dict)
        LiveLogger(metric_a, metric_b)
        number of training batches set: 3
        number of training epochs set: 1
        number of testing batches set: 0
        number of testing epochs set: 0
        """
        new = cls()
        # initializing LiveMetrics and setting data
        new._train_metrics.update(
            (key, LiveMetric.from_dict(metric))
            for key, metric in logger_dict["train_metrics"].items()
        )
        new._test_metrics.update(
            (key, LiveMetric.from_dict(metric))
            for key, metric in logger_dict["test_metrics"].items()
        )

        for train_mode, stat_mode in product(["train", "test"], ["batch", "epoch"]):
            item = "num_{}_{}".format(train_mode, stat_mode)
            setattr(new, "_" + item, logger_dict[item])
        return new

    def __init__(self, *args, **kwargs):
        """LiveLogger.__init__ does not utilize any input arguments, but accepts
        ``*args, **kwargs`` so that it can be used as a drop-in replacement for
         :obj:`~noggin.plotter.LivePlot`.
        """
        self._num_train_epoch = 0  # int: Current number of epochs trained
        self._num_train_batch = 0  # int: Current number of batches trained
        self._num_test_epoch = 0  # int: Current number of epochs tested
        self._num_test_batch = 0  # int: Current number of batches tested

        # stores batch/epoch-level training statistics and plot objects for training/testing
        self._train_metrics = OrderedDict()  # type: Dict[str, LiveMetric]
        self._test_metrics = OrderedDict()  # type: Dict[str, LiveMetric]

    def __repr__(self) -> str:
        metrics = sorted(set(self._train_metrics).union(set(self._test_metrics)))
        msg = "{}({})\n".format(type(self).__name__, ", ".join(metrics))

        words = (
            "training batches",
            "training epochs",
            "testing batches",
            "testing epochs",
        )
        things = (
            self._num_train_batch,
            self._num_train_epoch,
            self._num_test_batch,
            self._num_test_epoch,
        )

        for word, thing in zip(words, things):
            msg += "number of {word} set: {thing}\n".format(word=word, thing=thing)

        return msg

    def set_train_batch(self, metrics: Dict[str, Real], batch_size: Integral, **kwargs):
        """Record batch-level measurements for train-metrics.

        Parameters
        ----------
        metrics : Dict[str, Real]
            Mapping of metric-name to value. Only those metrics that were
            registered when initializing LivePlot will be recorded.

        batch_size : Integral
            The number of samples in the batch used to produce the metrics.
            Used to weight the metrics to produce epoch-level statistics.

        Notes
        -----
        ``**kwargs`` is included in the signature only to facilitate a seamless
        drop-in replacement for :obj:`~noggin.plotter.LivePlot`. It is not
        utilized here.
        """

        if not self._num_train_batch:
            # initialize batch-level metrics
            self._train_metrics.update((key, LiveMetric(key)) for key in metrics)

        # record each incoming batch metric
        for key, value in metrics.items():
            try:
                self._train_metrics[key].add_datapoint(value, weighting=batch_size)
            except KeyError:
                pass

        self._num_train_batch += 1

    def set_train_epoch(self):
        """Record an epoch for the train-metrics.

        Computes epoch-level statistics based on the batches accumulated since
        the prior epoch.
        """
        # compute epoch-mean metrics
        for key in self._train_metrics:
            self._train_metrics[key].set_epoch_datapoint()

        self._num_train_epoch += 1

    def set_test_batch(self, metrics: Dict[str, Real], batch_size: Integral):
        """Record batch-level measurements for test-metrics.

        Parameters
        ----------
        metrics : Dict[str, Real]
            Mapping of metric-name to value. Only those metrics that were
            registered when initializing LivePlot will be recorded.

        batch_size : Integral
            The number of samples in the batch used to produce the metrics.
            Used to weight the metrics to produce epoch-level statistics.
        """
        if not self._num_test_batch:
            self._test_metrics.update((key, LiveMetric(key)) for key in metrics)

        # record each incoming batch metric
        for key, value in metrics.items():
            try:
                self._test_metrics[key].add_datapoint(value, weighting=batch_size)
            except KeyError:
                pass

        self._num_test_batch += 1

    def set_test_epoch(self):
        """Record an epoch for the test-metrics.

        Computes epoch-level statistics based on the batches accumulated since
        the prior epoch.
        """
        # compute epoch-mean metrics
        for key in self._test_metrics:
            try:
                x = (
                    self._train_metrics[key].batch_domain[-1]
                    if self._train_metrics
                    else None
                )
            except (KeyError, IndexError):
                x = None
            self._test_metrics[key].set_epoch_datapoint(x)

        self._num_test_epoch += 1
