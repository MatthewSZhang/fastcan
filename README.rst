FastCan: A Fast Canonical-Correlation-Based Feature Selection Method
====================================================================

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
