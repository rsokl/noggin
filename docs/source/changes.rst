=========
Changelog
=========

This is a record of all past noggin releases and what went into them,
in reverse chronological order. All previous releases should still be available
on pip.

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