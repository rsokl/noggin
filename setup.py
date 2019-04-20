from setuptools import setup, find_packages


def do_setup():
    setup(
        name="LivePlot",
        version="0.1",
        author="Ryan Soklaski",
        license="MIT",
        platforms=["Windows", "Linux", "Mac OS-X", "Unix"],
        packages=find_packages(),
        install_requires=["matplotlib>=1.5"],
    )


if __name__ == "__main__":
    do_setup()
