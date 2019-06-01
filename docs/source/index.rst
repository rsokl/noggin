.. Noggin documentation master file, created by
   sphinx-quickstart on Thu May 30 11:24:42 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Noggin
======

.. image:: _static/liveplot.gif

Noggin is a python simple tool for 'live' logging and plotting measurements during experiments. Although Noggin can be used in a general context, it is designed around the train/test and batch/epoch paradigm for training a machine learning model.

Noggin's primary features are its abilities to:

- Log batch-level and epoch-level measurements by name
- Seamlessly update a 'live' plot of your measurements, embedded within a `Jupyter notebook <https://www.pythonlikeyoumeanit.com/Module1_GettingStartedWithPython/Jupyter_Notebooks.html>`_
- Organize your measurements into a data set of arrays with labeled axes, via `xarray <http://xarray.pydata.org/en/stable/index.html>`_
- Save and load your measurements & live-plot session: resume your experiment later without a hitch


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   workflow
   documentation
   changes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
