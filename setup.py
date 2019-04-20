from setuptools import setup, find_packages


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
    )


if __name__ == "__main__":
    do_setup()
