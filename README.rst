FastCan: A Fast Canonical-Correlation-Based Feature Selection Method
====================================================================
|Codecov| |CI| |Doc| |PythonVersion| |PyPi| |Black| |ruff| |pixi|


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




FastCan is a Python implementation of the following papers.

#. Zhang, S., & Lang, Z. Q. (2022).
    Orthogonal least squares based fast feature selection for
    linear classification. Pattern Recognition, 123, 108419.

#. Zhang, S., Wang, T., Sun L., Worden, K., & Cross, E. J. (2024).
    Canonical-correlation-based fast feature selection for
    structural health monitoring.

Installation
------------

Install **FastCan**:

* Run ``pip install fastcan``

Examples
--------
>>> from fastcan import FastCan
>>> X = [[ 0.87, -1.34,  0.31 ],
...     [-2.79, -0.02, -0.85 ],
...     [-1.34, -0.48, -2.55 ],
...     [ 1.92,  1.48,  0.65 ]]
>>> y = [0, 1, 0, 1]
>>> selector = FastCan(n_features_to_select=2, verbose=0).fit(X, y)
>>> selector.get_support()
array([ True,  True, False])

Uninstallation
--------------
Uninstall **FastCan**:

* Run ``pip uninstall fastcan``
