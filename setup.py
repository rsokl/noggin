from setuptools import find_packages, setup


def do_setup():
    import versioneer

    setup(
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        name="LivePlot",
        author="Ryan Soklaski",
        license="MIT",
        platforms=["Windows", "Linux", "Mac OS-X", "Unix"],
        package_dir={"": "src"},
        packages=find_packages(where="src", exclude=["tests", "tests.*"]),
        python_requires=">=3.6",
        install_requires=["numpy>=1.12", "matplotlib>=2.0", "xarray>=0.1"],
    )


if __name__ == "__main__":
    do_setup()
