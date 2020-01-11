# noggin

![Python version support](https://img.shields.io/badge/python-3.6%20&#8208;%203.8-blue.svg)
[![PyPi version](https://img.shields.io/pypi/v/noggin.svg)](https://pypi.python.org/pypi/noggin)
[![Build Status](https://travis-ci.com/rsokl/noggin.svg?branch=master)](https://travis-ci.com/rsokl/noggin)
[![codecov](https://codecov.io/gh/rsokl/noggin/branch/master/graph/badge.svg)](https://codecov.io/gh/rsokl/noggin)
[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)
[![Documentation Status](https://readthedocs.org/projects/noggin/badge/?version=latest)](https://noggin.readthedocs.io/en/latest/?badge=latest)

Noggin is a simple Python tool for ‘live’ logging and plotting measurements during experiments. Although Noggin can be used in a general context, it is designed around the train/test and batch/epoch paradigm for training a machine learning model.

Noggin’s primary features are its abilities to:

- Log batch-level and epoch-level measurements by name
- Seamlessly update a ‘live’ plot of your measurements, embedded within a Jupyter notebook
- Organize your measurements into a data set of arrays with labeled axes, via xarray
- Save and load your measurements & live-plot session: resume your experiment later without a hitch

You can read mode about Noggin [here](https://noggin.readthedocs.io/en/latest)

![noggin](https://user-images.githubusercontent.com/29104956/52166468-bf425700-26db-11e9-9324-1fc83d4bc71d.gif)
