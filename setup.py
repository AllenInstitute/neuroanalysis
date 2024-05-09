from os import path

from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    author="Luke Campagnola",
    author_email="lukec@alleninstitute.org",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    description="Functions and modular user interface tools for analysis of patch clamp experiment data.",
    extra_require={
        "ACQ4": ["acq4"],
        "jit": ["numba"],
        "MIES": ["h5py"],
        "test": ["pytest"],
        "ui": ["pyqtgraph"],
    },
    install_requires=["lmfit", "numpy", "scipy"],
    keywords="neuroscience analysis neurodata without borders nwb",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="neuroanalysis",
    packages=find_packages(),
    url="https://github.com/AllenInstitute/neuroanalysis",
    version="0.0.2",
)
