FastCan: A Fast Canonical-Correlation-Based Feature Selection Algorithm
=======================================================================
|conda| |Codecov| |CI| |Doc| |PythonVersion| |PyPi| |Black| |ruff| |pixi|

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/fastcan.svg
   :target: https://anaconda.org/conda-forge/fastcan

.. |Codecov| image:: https://codecov.io/gh/MatthewSZhang/fastcan/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/MatthewSZhang/fastcan

.. |CI| image:: https://github.com/MatthewSZhang/fastcan/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/MatthewSZhang/fastcan/actions

.. |Doc| image:: https://readthedocs.org/projects/fastcan/badge/?version=latest
   :target: https://fastcan.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/fastcan.svg
   :target: https://pypi.org/project/fastcan/

.. |PyPi| image:: https://img.shields.io/pypi/v/fastcan
   :target: https://pypi.org/project/fastcan

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff

.. |pixi| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
   :target: https://pixi.sh

FastCan is a feature selection method, which has following advantages:

#. Extremely **fast**.

#. Support unsupervised feature selection.

#. Support multioutput feature selection.

#. Skip redundant features.

#. Evalaute relative usefulness of features.

Check `Home Page <https://fastcan.readthedocs.io/en/latest/?badge=latest>`_ for more information.

Installation
------------

Install **FastCan** via PyPi:

* Run ``pip install fastcan``

Or via conda-forge:

* Run ``conda install -c conda-forge fastcan``

Getting Started
---------------
>>> from fastcan import FastCan
>>> X = [[1, 0], [0, 1]]
>>> y = [1, 0]
>>> FastCan(verbose=0).fit(X, y).get_support()
array([ True, False])


Citation
--------

FastCan is a Python implementation of the following papers.

If you use the `h-correlation` method in your work please cite the following reference:

.. code:: bibtex

   @article{ZHANG2022108419,
      title = {Orthogonal least squares based fast feature selection for linear classification},
      journal = {Pattern Recognition},
      volume = {123},
      pages = {108419},
      year = {2022},
      issn = {0031-3203},
      doi = {https://doi.org/10.1016/j.patcog.2021.108419},
      url = {https://www.sciencedirect.com/science/article/pii/S0031320321005951},
      author = {Sikai Zhang and Zi-Qiang Lang},
      keywords = {Feature selection, Orthogonal least squares, Canonical correlation analysis, Linear discriminant analysis, Multi-label, Multivariate time series, Feature interaction},
      }

If you use the `eta-cosine` method in your work please cite the following reference:

.. code:: bibtex

   @article{ZHANG2025111895,
      title = {Canonical-correlation-based fast feature selection for structural health monitoring},
      journal = {Mechanical Systems and Signal Processing},
      volume = {223},
      pages = {111895},
      year = {2025},
      issn = {0888-3270},
      doi = {https://doi.org/10.1016/j.ymssp.2024.111895},
      url = {https://www.sciencedirect.com/science/article/pii/S0888327024007933},
      author = {Sikai Zhang and Tingna Wang and Keith Worden and Limin Sun and Elizabeth J. Cross},
      keywords = {Multivariate feature selection, Filter method, Canonical correlation analysis, Feature interaction, Feature redundancy, Structural health monitoring},
      }
