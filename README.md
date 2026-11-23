## Automatic Structure Phase Map (autophasemap) generation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24484864.svg)](https://doi.org/10.5281/zenodo.24484864)

This repository hosts the supporting code for the following paper.
[Metric geometry tools for automatic structure phase map generation](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00105a) by Kiran Vaddi, Karen Li, and Lilo D Pozzo.

To install the package (Python 3.10 or newer):

```bash
pip install git+https://github.com/pozzo-research-group/autophasemap.git 
```

This also installs the [warping](https://github.com/kiranvad/warping) dependency, which compiles a C extension, so a C compiler is required (e.g. Xcode command line tools on macOS). See [INSTALL.md](INSTALL.md) for development and conda installs.

A simple example using synthetic Gaussian functions as an example can be found in [this ](/example.ipynb)notebook. This example can be run on your local machine once the `autophasemap` package is installed using similar instructions as above. The same example runs as an automated test in [tests/test_example.py](tests/test_example.py).


A full list of examples from the original paper above can be found [here](https://github.com/pozzo-research-group/papers/tree/main/autophasemap)
