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
    def name(self):
        return self._name

    @property
    def batch_domain(self) -> ndarray:
        return np.arange(1, len(self.batch_data) + 1, dtype=float)

    @property
    def batch_data(self) -> ndarray:
        """
        Metric data for consecutive batches.

        Returns
        -------
        numpy.ndarray, shape=(N_batch, )"""
        if self._batch_data_list:
            self._batch_data = np.concatenate((self._batch_data, self._batch_data_list))
            self._batch_data_list = []
        return self._batch_data

    @property
    def epoch_domain(self) -> ndarray:
        if self._epoch_domain_list:
            self._epoch_domain = np.concatenate(
                (self._epoch_domain, self._epoch_domain_list)
            )
            self._epoch_domain_list = []
        return self._epoch_domain

    @property
    def epoch_data(self) -> ndarray:
        """
        Metric data for consecutive epochs.

        Returns
        -------
        numpy.ndarray, shape=(N_epoch, )"""
        if self._epoch_data_list:
            self._epoch_data = np.concatenate((self._epoch_data, self._epoch_data_list))
            self._epoch_data_list = []
        return self._epoch_data

    def add_datapoint(self, value: Real, weighting: Real = 1.0):
        """
        Parameters
        ----------
        value : Real
        weighting : Real """
        if isinstance(value, np.ndarray):
            value = value.item()

        self._batch_data_list.append(value)
        self._running_weighted_sum += weighting * value
        self._total_weighting += weighting
        self._cnt_since_epoch += 1

    def set_epoch_datapoint(self, x: Optional[Real] = None):
        """ Parameters
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
        The encoded dictionary stores:

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
        """ The inverse of `LiveMetric.to_dict`. Given a dictionary of
        live-metric data, constructs an instance of `LiveMetric`.

        Parameters
        ----------
        metrics_dict: Dict[str, ndarray]
            Stores the state of the live-metric instance being created.

        Returns
        -------
        LiveMetric

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
    """

    @property
    def train_metrics(self) -> Dict[str, Dict[str, ndarray]]:
        """
        The batch and epoch data for each metric.

        Returns
        -------
        OrderedDict[str, Dict[str, numpy.ndarray]]

       '<metric-name>' -> {"batch_data":   array,
                           "epoch_data":   array,
                           "epoch_domain": array,
                           ...} """
        return OrderedDict((k, v.to_dict()) for k, v in self._train_metrics.items())

    @property
    def test_metrics(self) -> Dict[str, Dict[str, ndarray]]:
        """
        The batch and epoch data for each metric.

        Returns
        -------
        OrderedDict[str, Dict[str, numpy.ndarray]]

       '<metric-name>' -> {"batch_data":   array,
                           "epoch_data":   array,
                           "epoch_domain": array,
                           ...} """
        return OrderedDict((k, v.to_dict()) for k, v in self._test_metrics.items())

    def to_xarray(self, train_or_test: str) -> Tuple[Dataset, Dataset]:
        """
        Given noggin metrics, returns xarray datasets for the batch-level and epoch-level
        metrics, respectively.

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
        """
        from .xarray import metrics_to_xarrays

        if train_or_test not in ["train", "test"]:
            raise ValueError(
                "`train_or_test` must be 'train' or 'test',"
                "\nGot: {}".format(train_or_test)
            )
        metrics = self.train_metrics if train_or_test == "train" else self.test_metrics
        return metrics_to_xarrays(metrics)

    def to_dict(self):
        return dict(
            train_metrics=self.train_metrics,
            test_metrics=self.test_metrics,
            num_train_epoch=self._num_train_epoch,
            num_train_batch=self._num_train_batch,
            num_test_batch=self._num_test_batch,
            num_test_epoch=self._num_test_epoch,
        )

    @classmethod
    def from_dict(cls, logger_dict):
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

    def __init__(self):
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

    def set_train_batch(
        self, metrics: Dict[str, Real], batch_size: Integral
    ):  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Parameters
        ----------
        metrics : Dict[str, Real]
            Mapping of metric-name to value. Only those metrics that were
            registered when initializing LivePlot will be recorded.

        batch_size : Integral
            The number of samples in the batch used to produce the metrics.
            Used to weight the metrics to produce epoch-level statistics. """

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
        """
        Compute epoch-level statistics based on the batches accumulated since
        the prior batch.
        """
        # compute epoch-mean metrics
        for key in self._train_metrics:
            self._train_metrics[key].set_epoch_datapoint()

        self._num_train_epoch += 1

    def set_test_batch(self, metrics: Dict[str, Real], batch_size: Integral):
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
