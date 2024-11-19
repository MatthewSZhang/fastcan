"""
Extend feature selection
"""

import math
from copy import deepcopy
from numbers import Integral

import numpy as np
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_is_fitted

from ._cancorr_fast import _forward_search  # type: ignore
from ._fastcan import FastCan, _prepare_search


@validate_params(
    {
        "selector": [FastCan],
        "n_features_to_select": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "batch_size": [
            Interval(Integral, 1, None, closed="left"),
        ],
    },
    prefer_skip_nested_validation=False,
)
def extend(selector, n_features_to_select=1, batch_size=1):
    """Extend FastCan with mini batches.

    It is suitable for selecting a very large number of features
    even larger than the number of samples.

    Similar to the correlation filter which selects each feature without considering
    the redundancy, the function selects features in mini-batch and the
    redundancy between the two mini-batches will be ignored.

    Parameters
    ----------
    selector : FastCan
        FastCan selector.

    n_features_to_select : int, default=1
        The parameter is the absolute number of features to select.

    batch_size : int, default=1
        The number of features in a mini-batch.

    Returns
    -------
    indices : ndarray of shape (n_features_to_select,), dtype=int
        The indices of the selected features.

    Examples
    --------
    >>> from fastcan import FastCan, extend
    >>> X = [[1, 1, 0], [0.01, 0, 0], [-1, 0, 1], [0, 0, 0]]
    >>> y = [1, 0, -1, 0]
    >>> selector = FastCan(1, verbose=0).fit(X, y)
    >>> print(f"Indices: {selector.indices_}")
    Indices: [0]
    >>> indices = extend(selector, 3, batch_size=2)
    >>> print(f"Indices: {indices}")
    Indices: [0 2 1]
    """
    check_is_fitted(selector)
    n_inclusions = selector.indices_include_.size
    n_features = selector.n_features_in_
    n_to_select = n_features_to_select - selector.n_features_to_select
    batch_size_to_select = batch_size - n_inclusions

    if n_features_to_select > n_features:
        raise ValueError(
            f"n_features_to_select {n_features_to_select} "
            f"must be <= n_features {n_features}."
        )
    if n_to_select <= 0:
        raise ValueError(
            f"The number of features to select ({n_to_select}) ", "is less than 0."
        )
    if batch_size_to_select <= 0:
        raise ValueError(
            "The size of mini batch without included indices ",
            f"({batch_size_to_select}) is less than 0.",
        )

    X_transformed_ = deepcopy(selector.X_transformed_)

    indices_include = selector.indices_include_
    indices_exclude = selector.indices_exclude_
    indices_select = selector.indices_[n_inclusions:]

    n_threads = _openmp_effective_n_threads()

    for i in range(math.ceil(n_to_select / batch_size_to_select)):
        if i == 0:
            batch_size_i = (n_to_select - 1) % batch_size_to_select + 1 + n_inclusions
        else:
            batch_size_i = batch_size
        indices, scores, mask = _prepare_search(
            n_features,
            batch_size_i,
            indices_include,
            np.r_[indices_exclude, indices_select],
        )
        _forward_search(
            X=X_transformed_,
            V=selector.y_transformed_,
            t=batch_size_i,
            tol=selector.tol,
            num_threads=n_threads,
            verbose=0,
            mask=mask,
            indices=indices,
            scores=scores,
        )
        indices_select = np.r_[indices_select, indices[n_inclusions:]]
    return np.r_[indices_include, indices_select]
