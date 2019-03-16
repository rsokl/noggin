from collections import defaultdict
from typing import Dict, Sequence, Tuple

import numpy as np

import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from liveplot.typing import LiveMetrics
from liveplot.plotter import LiveLogger, LivePlot
from itertools import combinations

import pprint


__all__ = ["finite_arrays", "choices", "metric_dict", "live_metrics", "loggers"]


def finite_arrays(size):
    return hnp.arrays(
        shape=(size,),
        dtype=np.float64,
        elements=st.floats(allow_infinity=False, allow_nan=False),
    )


def choices(seq: Sequence, size: int) -> st.SearchStrategy[Tuple]:
    assert size <= len(seq)
    return st.sampled_from(tuple(combinations(seq, size)))


@st.composite
def metric_dict(draw, name,
                num_batch_data=None,
                num_epoch_data=None,
                epoch_domain=None) -> st.SearchStrategy[Dict[str, np.ndarray]]:
    if all(x is not None for x in (num_batch_data, num_epoch_data)):
        assert num_batch_data >= num_epoch_data

    if num_batch_data is None:
        num_batch_data = draw(st.integers(0, 5))

    if num_epoch_data is None:
        num_epoch_data = draw(st.integers(0, num_batch_data))

    if epoch_domain is None:
        epoch_domain = draw(choices(np.arange(1, num_batch_data + 1), size=num_epoch_data))

    out = dict(name=name)  # type: Dict[str, np.ndarray]
    out["batch_data"] = draw(finite_arrays(num_batch_data))  # type: np.ndarray
    out["epoch_data"] = draw(finite_arrays(num_epoch_data))  # type: np.ndarray
    out["epoch_domain"] = np.asarray(sorted(epoch_domain))
    out["cnt_since_epoch"] = draw(st.integers(0, num_batch_data - num_epoch_data))
    out["total_weighting"] = draw(st.floats(0., 10.)) if out["cnt_since_epoch"] else 0.
    out["running_weighted_sum"] = draw(st.floats(-10., 10.)) if out["cnt_since_epoch"] else 0.
    return out


@st.composite
def live_metrics(draw, min_num_metrics=0) -> st.SearchStrategy[LiveMetrics]:
    num_metrics = draw(st.integers(min_num_metrics, 3))
    num_batch_data = draw(st.integers(0, 5))
    num_epoch_data = draw(st.integers(0, num_batch_data))

    out = defaultdict(dict)  # type: Dict[str, Dict[str, np.ndarray]]
    for name in ["metric_a", "metric_b", "metric_c"][:num_metrics]:
        out[name] = draw(metric_dict(name,
                                     num_batch_data=num_batch_data,
                                     num_epoch_data=num_epoch_data,
                                     ))
    return dict(out.items())


def verbose_repr(self):
    metrics = sorted(set(self._train_metrics).union(set(self._test_metrics)))
    msg = "{}({})\n".format(type(self).__name__, ", ".join(metrics))

    words = ("training batches", "training epochs", "testing batches", "testing epochs")
    things = (self._num_train_batch, self._num_train_epoch,
              self._num_test_batch, self._num_test_epoch)

    for word, thing in zip(words, things):
        msg += "number of {word} set: {thing}\n".format(word=word, thing=thing)

    msg += "train metrics:\n{}\n".format(pprint.pformat(dict(self.train_metrics)))
    msg += "test metrics:\n{}".format(pprint.pformat(dict(self.test_metrics)))
    return msg


class VerboseLogger(LiveLogger):
    def __repr__(self): return verbose_repr(self)


class VerbosePlotter(LivePlot):
    def __repr__(self): return verbose_repr(self)


@st.composite
def loggers(draw) -> st.SearchStrategy[LiveLogger]:
    train_metrics = draw(live_metrics())
    test_metrics = draw(live_metrics())
    return VerboseLogger.from_dict(
        dict(train_metrics=train_metrics,
             test_metrics=test_metrics,
             num_train_epoch=max((len(v["epoch_data"]) for v in train_metrics.values()), default=0),
             num_train_batch=max((len(v["batch_data"]) for v in train_metrics.values()), default=0),
             num_test_epoch=max((len(v["epoch_data"]) for v in test_metrics.values()), default=0),
             num_test_batch=max((len(v["batch_data"]) for v in test_metrics.values()), default=0),
             )
    )


@st.composite
def plotters(draw) -> st.SearchStrategy[LivePlot]:
    train_metrics = draw(live_metrics())
    min_num_test = 1 if not train_metrics else 0
    test_metrics = draw(live_metrics(min_num_metrics=min_num_test))

    refresh = draw(st.one_of(st.just(-1), st.floats(0, 2)))
    metric_names = sorted(set(train_metrics).union(set(test_metrics)))
    nrows = len(metric_names)
    train_colors = {k: None for k in train_metrics}
    test_colors = {k: None for k in test_metrics}

    return LivePlot.from_dict(
        dict(train_metrics=train_metrics,
             test_metrics=test_metrics,
             num_train_epoch=max((len(v["epoch_data"]) for v in train_metrics.values()), default=0),
             num_train_batch=max((len(v["batch_data"]) for v in train_metrics.values()), default=0),
             num_test_epoch=max((len(v["epoch_data"]) for v in test_metrics.values()), default=0),
             num_test_batch=max((len(v["batch_data"]) for v in test_metrics.values()), default=0),
             refresh=refresh,
             pltkwargs=dict(figsize=(3, 2), nrows=nrows, ncols=1),
             train_colors=train_colors,
             test_colors=test_colors,
             metric_names=metric_names,
             )
    )
