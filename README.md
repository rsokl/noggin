[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rsokl/liveplot/master?filepath=LivePlot_Demo.ipynb)

# nogging
Log and plot metrics during train/test time for a neural network (or whatever, really). `nogging`
provides convenient i/o functions for saving, loading, and recreating nogging sessions. It also provides
an interface for accessing logged metrics as [xarray data sets](http://xarray.pydata.org/en/stable/index.html). This
functionality, availabile via `nogging.xarray`, permits users to seamlessly access their logged metrics as N-dimensional arrays with named axes.

Please consult the [demo notebook](https://github.com/rsokl/LivePlot/blob/master/LivePlot_Demo.ipynb) for a summary of `nogging`'s functionality.

![nogging](https://user-images.githubusercontent.com/29104956/52166468-bf425700-26db-11e9-9324-1fc83d4bc71d.gif)


## Installing nogging
Clone/download this repository, navigate to the `nogging` directory, and run:
```shell
python setup.py install
```

## Examples
### Creating a live plot
```python
import numpy as np
from nogging import create_plot

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


from nogging import save_metrics

# save metrics from nogging instance
save_metrics("tmp.npz", plotter)
```

