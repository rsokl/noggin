import versioneer
from setuptools import find_packages, setup

DISTNAME = "noggin"
LICENSE = "MIT"
AUTHOR = "Ryan Soklaski"
AUTHOR_EMAIL = "ryan.soklaski@gmail.com"
URL = "https://github.com/rsokl/noggin"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
]

INSTALL_REQUIRES = (
    ["numpy>=1.9", "matplotlib>=2.0", "xarray>=0.1", "custom_inherit>=2.2"],
)
TESTS_REQUIRE = ["pytest >= 3.8", "hypothesis >= 4.22.3"]

DESCRIPTION = "A simple tool for logging and plotting measurements when training a neural network."
LONG_DESCRIPTION = """
noggin is a simple tool for logging and plotting measurements during an experiment. In particular, it
was created as a convenient means for recording metrics while training a neural network, thus it is
designed around the familiar training/testing and batch/epoch paradigms.

noggin is able to perform "live plotting" and automatically update its plots throughout an experiment.
Its logging interface also makes it easy to save and load recorded metrics, and seamlessly resume your experiment.
These metrics can also be accessed as an N-dimension array with labeled axes, by making keen use of the ``xarray``
library.
"""


setup(
    name=DISTNAME,
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    url=URL,
    download_url="https://github.com/rsokl/noggin/tarball/" + versioneer.get_version(),
    python_requires=">=3.6",
)
