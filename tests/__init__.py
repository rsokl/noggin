from contextlib import contextmanager
import matplotlib.pyplot as plt


@contextmanager
def close_plots():
    try:
        yield None
    finally:
        # Code to release resource, e.g.:
        plt.close("all")
