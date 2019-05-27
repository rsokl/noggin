from contextlib import contextmanager
import matplotlib.pyplot as plt


@contextmanager
def close_plots():
    try:
        yield None
    finally:
        plt.close("all")
