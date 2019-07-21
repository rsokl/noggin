=========
Changelog
=========

This is a record of all past noggin releases and what went into them,
in reverse chronological order. All previous releases should still be available
on pip.

.. _v0.10.1:

-------------------
0.10.1 - 2019-07-21
-------------------

Fixes bug which `last_n_batches` was specified for a :class:`~noggin.plotter.LivePlot` instance, and a
metric's test-epochs were being plotted, but its train-epochs were not. In this scenario, `noggin` was not
properly tracking the batch-iterations associated with the plotted epochs, and *all* of the test-epochs were
being plotted.

.. _v0.10.0:

-------------------
0.10.0 - 2019-06-15
-------------------

Normalizes the interfaces of :class:`~noggin.logger.LiveLogger` and :class:`~noggin.plotter.LivePlot`
so that they can be used as drop-in replacements for each other more seamlessly.

This is an API-breaking update for :class:`~noggin.plotter.LivePlot`, as it renames the methods
``plot_train_epoch`` and ``plot_test_epoch`` to ``set_train_epoch`` and ``set_test_epoch``,
respectively. As stated above, this is to match the interface of  :class:`~noggin.logger.LiveLogger`.


.. _v0.9.1:

-------------------
0.9.1 - 2019-06-06
-------------------

Adds :func:`~noggin.utils.plot_logger`, which provides a convenient means for plotting
the data stored by a :class:`~noggin.logger.LiveLogger`, and to convert it into an
instance of :class:`~noggin.plotter.LivePlot`

:class:`~noggin.plotter.LivePlot` not longer warns about a bad matplotlib backend
if ``max_fraction_spent_plotting`` is set to ``0.``


.. _v0.9.0:

-------------------
0.9.0 - 2019-05-27
-------------------

This is the first public release of noggin on pypi.