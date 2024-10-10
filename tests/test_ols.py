"Test ols"

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.linear_model import LinearRegression

from fastcan import ols


def test_sum_errs():
    """Test multiple correlation."""
    rng = np.random.default_rng(12345)
    X = rng.random((100, 10))
    y = rng.random(100)

    indices, scores = ols(X, y, 5)

    y_hat = LinearRegression(fit_intercept=False)\
                .fit(X[:, indices], y)\
                .predict(X[:, indices])
    e = y-y_hat
    # Sum of Error Reduction Ratio
    serrs = 1 - np.dot(e, e)/np.dot(y, y)
    assert_almost_equal(actual=scores.sum(), desired=serrs)
