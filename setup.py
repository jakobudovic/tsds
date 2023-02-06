import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
setup(
    name="tsds",
    version="0.0.1",
    description=" A python library used to analyse any time series data",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Jakob Udovic",
    author_email="jakobudovic2@gmail.com",
    license="GNU",
    packages=["tsds"],
    zip_safe=False,
)
