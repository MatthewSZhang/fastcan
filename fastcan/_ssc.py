"""Sum squared of correlation."""

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.utils import check_X_y
from sklearn.utils._param_validation import validate_params


@validate_params(
    {
        "X": ["array-like"],
        "y": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def ssc(X, y):
    """Sum of the squared canonical correlation coefficients.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples, n_outputs)
        Target matrix.

    Returns
    -------
    ssc : float
        Sum of the squared canonical correlation coefficients.

    Examples
    --------
    >>> from fastcan import ssc
    >>> X = [[1], [-1], [0]]
    >>> y = [[0], [1], [-1]]
    >>> ssc(X, y)
    np.float64(0.25)
    """
    X, y = check_X_y(
        X, y, dtype=float, ensure_2d=True, multi_output=True, ensure_min_samples=2
    )
    n_components = min(X.shape[1], y.shape[1])
    cca = CCA(n_components=n_components)
    X_c, y_c = cca.fit_transform(X, y)
    corrcoef = np.diagonal(np.corrcoef(X_c, y_c, rowvar=False), offset=n_components)
    return sum(corrcoef**2)
