# noggin

![Python version support](https://img.shields.io/badge/python-3.6%20%203.7-blue.svg)
[![PyPi version](https://img.shields.io/pypi/v/noggin.svg)](https://pypi.python.org/pypi/noggin)
[![Build Status](https://travis-ci.com/rsokl/noggin.svg?branch=master)](https://travis-ci.com/rsokl/noggin)
[![codecov](https://codecov.io/gh/rsokl/noggin/branch/master/graph/badge.svg)](https://codecov.io/gh/rsokl/noggin)
[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)
[![Documentation Status](https://readthedocs.org/projects/noggin/badge/?version=latest)](https://noggin.readthedocs.io/en/latest/?badge=latest)

Log and plot metrics during train/test time for a neural network (or whatever, really). `noggin`
provides convenient i/o functions for saving, loading, and recreating noggin sessions. It also provides
an interface for accessing logged metrics as [xarray data sets](http://xarray.pydata.org/en/stable/index.html). This
functionality, available via `noggin.xarray`, permits users to seamlessly access their logged metrics as N-dimensional arrays with named axes.


![noggin](https://user-images.githubusercontent.com/29104956/52166468-bf425700-26db-11e9-9324-1fc83d4bc71d.gif)


## Installing noggin
Clone/download this repository, navigate to the `noggin` directory, and run:
```shell
pip install noggin
```

## Examples
### Creating a live plot
```python
import numpy as np
from noggin import create_plot

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


from noggin import save_metrics

# save metrics from noggin instance
save_metrics("tmp.npz", plotter)
```

