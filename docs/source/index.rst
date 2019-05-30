.. noggin documentation master file, created by
   sphinx-quickstart on Thu May 30 11:24:42 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

noggin
======
noggin is a python simple tool for 'live' logging and plotting measurements during experiments.

.. image:: _static/liveplot.gif

Although noggin can be used in a general context, it is designed around the train/test and batch/epoch paradigm for
training a neural network.

noggin's primary features are its abilities to:

- Log batch-level and epoch-level measurements by name
- Seamlessly update a 'live' plot of your measurements, embedded within a `Jupyter notebook <https://www.pythonlikeyoumeanit.com/Module1_GettingStartedWithPython/Jupyter_Notebooks.html>`_
- Organize your measurements into a data set of labelled arrays, via `xarray <http://xarray.pydata.org/en/stable/index.html>`_
- Save and load your measurements & live-plot: resume your experiment later without a hitch


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
