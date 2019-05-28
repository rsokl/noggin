import os
import tempfile

import matplotlib.pyplot as plt
import pytest
from hypothesis import Verbosity, settings

settings.register_profile("ci", deadline=None)
settings.register_profile("intense", deadline=None, max_examples=1000)
settings.register_profile("dev", max_examples=10)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture()
def cleandir() -> str:
    """ This fixture will use the stdlib `tempfile` module to
    move the current working directory to a tmp-dir for the
    duration of the test.

    Afterwards, the session returns to its previous working
    directory, and the temporary directory and its contents
    are removed.

    Yields
    ------
    str
        The name of the temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_dir = os.getcwd()
        os.chdir(tmpdirname)
        yield tmpdirname
        os.chdir(old_dir)


@pytest.fixture(scope="session", autouse=True)
def killplots():
    """Ensures all matplotlib figures are closed upon leaving the fixture"""
    yield None
    plt.close("all")
