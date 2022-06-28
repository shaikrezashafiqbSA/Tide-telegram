from setuptools import setup, find_packages

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name="The Tides",
    version="0.0.1",
    description="Trend Following model",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    author="alphastone",
    author_email="",
    license="",
    classifiers=[
        "Programming Language :: Python :: 3.9.13",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numba",
        "tqdm",
        "matplotlib",
        "plotly",
        "pandas_ta",
        "plotly_resampler"
        "tvdatafeed",
        "python-telegram-bot",
        "dataframe_image",
        "requests"
    ]
)