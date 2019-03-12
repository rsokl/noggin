"""
Provides functionality for logging training and testing batch-level & epoch-level metrics
"""

from numbers import Integral, Real
import numpy as np
from collections import OrderedDict

from numpy import ndarray
from typing import Dict, Optional, Tuple

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
            raise TypeError("Metric names must be specified as strings. Got: {}".format(name))
        self.name = name
        self.batch_line = None  # ax object for batch data
        self.epoch_line = None  # ax object for epoch data
        self._batch_data = []  # metric data for consecutive batches
        self._epoch_data = []  # accuracies
        self._epoch_domain = []
        self._running_weighted_sum = 0.
        self._total_weighting = 0.
        self._cnt_since_epoch = 0

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
        return np.array(self._batch_data)

    @property
    def epoch_domain(self) -> ndarray:
        return np.array(self._epoch_domain)

    @property
    def epoch_data(self) -> ndarray:
        """
        Metric data for consecutive epochs.

        Returns
        -------
        numpy.ndarray, shape=(N_epoch, )"""
        return np.array(self._epoch_data)

    def add_datapoint(self, value: Real, weighting: Real = 1.):
        """
        Parameters
        ----------
        value : Real
        weighting : Real """
        if isinstance(value, np.ndarray):
            value = np.asscalar(value)

        self._batch_data.append(value)
        self._running_weighted_sum += weighting * value
        self._total_weighting += weighting
        self._cnt_since_epoch += 1

    def set_epoch_datapoint(self, x: Optional[Real] = None):
        """ Parameters
            ----------
            x : Optional[Real]
                Specify the domain-value to be set for this data point."""
        if self._cnt_since_epoch > 0:
            mean = self._running_weighted_sum / (self._total_weighting
                                                 if self._total_weighting > 0.
                                                 else 1.)
            self._epoch_data.append(mean)
            self._epoch_domain.append(x if x is not None else self.batch_domain[-1])
            self._running_weighted_sum = 0.
            self._total_weighting = 0.
            self._cnt_since_epoch = 0

    def to_dict(self) -> Dict[str, ndarray]:
        """
        Returns the batch data, epoch domain, and epoch data
        in a dictionary.

        Additionally, running statistics are included in order to
        preserve the state of the metric.

        Returns
        -------
        Dict[str, ndarray]
        """
        out = {attr: getattr(self, attr)
               for attr in ("batch_data", "epoch_data", "epoch_domain")}
        out.update((attr, getattr(self, "_" + attr))
                   for attr in ("cnt_since_epoch", "total_weighting", "running_weighted_sum"))
        return out

    @classmethod
    def from_dict(cls, name: str, metrics_dict: Dict[str, ndarray]):
        array_keys = ("batch_data", "epoch_data", "epoch_domain")
        running_stats_keys = ("running_weighted_sum", "total_weighting", "cnt_since_epoch")
        required_keys = array_keys + running_stats_keys

        if not isinstance(metrics_dict, dict):
            raise TypeError(
                "`live_metrics` must be a dictionary, "
                "got type {}".format(type(metrics_dict))
            )

        if not set(required_keys) <= set(metrics_dict):
            raise ValueError(
                "`live_metrics` is missing the following keys: "
                "'{}'".format(", ".join(set(required_keys) - set(metrics_dict)))
            )

        out = cls(name)
        for k in required_keys:
            v = metrics_dict[k]
            if k in array_keys:
                if not isinstance(v, np.ndarray) and v.ndim == 1:
                    raise ValueError(
                        "'{}' must map to a 1D numpy arrays".format(k)
                    )
                setattr(out, "_" + k, v.tolist())
            else:
                if not isinstance(v, Real):
                    raise ValueError(
                        "'{}' must map to a real number".format(k)
                    )
                if k == "cnt_since_epoch" and (not isinstance(v, Integral) or v < 0):
                    raise ValueError(
                        "{} must map to a non-negative value".format(k)
                    )
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
        out = OrderedDict()
        for k, v in self._train_metrics.items():
            out[k] = v.to_dict()
        return out

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
        out = OrderedDict()
        for k, v in self._test_metrics.items():
            out[k] = v.to_dict()
        return out

    def __init__(self):
        self._metrics = tuple()  # type: Tuple[str, ...]

        self._train_epoch_num = 0  # int: Current number of epochs trained
        self._train_batch_num = 0  # int: Current number of batches trained
        self._test_epoch_num = 0  # int: Current number of epochs tested
        self._test_batch_num = 0  # int: Current number of batches tested

        # stores batch/epoch-level training statistics and plot objects for training/testing
        self._train_metrics = OrderedDict()  # type: Dict[str, LiveMetric]
        self._test_metrics = OrderedDict()  # type: Dict[str, LiveMetric]

    def __repr__(self) -> str:
        msg = "{}({})\n\n".format(type(self).__name__, ", ".join(self._metrics))

        words = ("training batches", "training epochs", "testing batches", "testing epochs")
        things = (self._train_batch_num, self._train_epoch_num,
                  self._test_batch_num, self._test_epoch_num)

        for word, thing in zip(words, things):
            msg += "number of {word} set: {thing}\n".format(word=word, thing=thing)

        return msg

    def set_train_batch(self, metrics: Dict[str, Real], batch_size: Integral):
        """
        Parameters
        ----------
        metrics : Dict[str, Real]
            Mapping of metric-name to value. Only those metrics that were
            registered when initializing LivePlot will be recorded.

        batch_size : Integral
            The number of samples in the batch used to produce the metrics.
            Used to weight the metrics to produce epoch-level statistics. """

        if not self._train_batch_num:
            # initialize batch-level metrics
            self._train_metrics.update((key, LiveMetric(key)) for key in metrics)

        # record each incoming batch metric
        for key, value in metrics.items():
            try:
                self._train_metrics[key].add_datapoint(value, weighting=batch_size)
            except KeyError:
                pass

        self._train_batch_num += 1

    def set_train_epoch(self):
        """
        Compute epoch-level statistics based on the batches accumulated since
        the prior batch.
        """
        # compute epoch-mean metrics
        for key in self._train_metrics:
            self._train_metrics[key].set_epoch_datapoint()

        self._train_epoch_num += 1

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
        if not self._test_batch_num:
            self._test_metrics.update((key, LiveMetric(key)) for key in metrics)

        # record each incoming batch metric
        for key, value in metrics.items():
            try:
                self._test_metrics[key].add_datapoint(value, weighting=batch_size)
            except KeyError:
                pass

        self._test_batch_num += 1

    def set_test_epoch(self):
        # compute epoch-mean metrics
        for key in self._test_metrics:
            try:
                x = self._train_metrics[key].batch_domain[-1] if self._train_metrics else None
            except KeyError:
                x = None
            self._test_metrics[key].set_epoch_datapoint(x)

        self._test_epoch_num += 1

    # @classmethod
    # def from_metrics(cls,
    #                  train_metrics: Optional[Dict[str, Dict[str, ndarray]]],
    #                  test_metrics: Optional[Dict[str, Dict[str, ndarray]]]):
    #     if train_metrics is None:
    #         train_metrics = {}
    #
    #     if test_metrics is None:
    #         test_metrics = {}
    #
    #     new = cls()
    #     # initializing LiveMetrics and setting data
    #     new._train_metrics.update((key, LiveMetric.from_dict(key, metric))
    #                               for key, metric in train_metrics.items()
    #                               if key in new._metrics)
    #     new._test_metrics.update((key, LiveMetric.from_dict(key, metric))
    #                              for key, metric in test_metrics.items()
    #                              if key in new._metrics)
    #
    #     # setting num_batch/num_epoch values
    #     num_name = ["_train_epoch_num", "_train_batch_num", "_test_epoch_num", "_test_batch_num"]
    #     vals = [0, 0, 0, 0]
    #
    #     if train_metrics:
    #         vals[0] = max(len(v["epoch_data"]) for v in train_metrics.values())
    #         vals[1] = max(len(v["batch_data"]) for v in train_metrics.values())
    #     if test_metrics:
    #         vals[2] = max(len(v["epoch_data"]) for v in test_metrics.values())
    #         vals[3] = max(len(v["batch_data"]) for v in test_metrics.values())
    #
    #     for name, val in zip(num_name, vals):
    #         setattr(new, name, val)
    #     return new


