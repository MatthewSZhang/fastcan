"Test ssc"

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.linear_model import LinearRegression

from fastcan import ssc


def test_pearson_r():
    """Test Pearson's correlation."""
    rng = np.random.default_rng(12345)
    X = rng.random(100)
    y = rng.random(100)
    r2 = ssc(X.reshape(-1, 1), y.reshape(-1, 1))
    gtruth_r2 = np.corrcoef(X, y)[0, 1]**2
    assert_almost_equal(actual=r2, desired=gtruth_r2)

def test_multi_r():
    """Test multiple correlation."""
    rng = np.random.default_rng(12345)
    X = rng.random((100, 10))
    y = rng.random(100)
    r2 = ssc(X, y.reshape(-1, 1))
    gtruth_r2 = LinearRegression().fit(X, y).score(X, y)
    assert_almost_equal(actual=r2, desired=gtruth_r2)

    X = rng.random(100)
    y = rng.random((100, 10))
    r2 = ssc(X.reshape(-1, 1), y)
    gtruth_r2 = LinearRegression().fit(y, X).score(y, X)
    assert_almost_equal(actual=r2, desired=gtruth_r2)
