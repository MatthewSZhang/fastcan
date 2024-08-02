# pylint: skip-file
"""Test FastCan"""

import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.cross_decomposition import CCA
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import check_estimator

from fastcan import FastCan


def test_fastcan_is_sklearn_estimator():
    check_estimator(FastCan())

def test_select_kbest_classif():
    # Test whether the relative univariate feature selection
    # gets the correct items in a simple classification problem
    # with the k best heuristic
    n_samples = 200
    n_features = 20
    n_classes = 8
    n_informative = 5

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )

    reg = LinearRegression().fit(X[:, :n_informative], y)
    gtruth_ssc = reg.score(X[:, :n_informative], y)

    correlation_filter = FastCan(
        n_features_to_select=n_informative,
    )
    correlation_filter.fit(X, y)
    ssc = correlation_filter.scores_.sum()
    # Test whether the ssc from the fastcan is consistent
    # with the mcc from the linear regression
    assert_almost_equal(actual=ssc, desired=gtruth_ssc)

    support = correlation_filter.get_support()
    gtruth = np.zeros(n_features)
    gtruth[:n_informative] = 1
    assert_array_equal(support, gtruth)

def test_inclusive_indices():
    # Test whether fastcan can select informative features based
    # on some pre-include features
    n_samples = 20
    n_features = 20
    n_targets = 8
    n_informative = 5

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        noise=0.1,
        shuffle=False,
        random_state=0,
    )

    correlation_filter = FastCan(
        n_features_to_select=n_informative,
        inclusive_indices=[0, 3]
    )
    correlation_filter.fit(X, y)

    support = correlation_filter.get_support()
    gtruth = np.zeros(n_features)
    gtruth[:n_informative] = 1
    assert_array_equal(support, gtruth)

def test_ssc_consistent_with_cca():
    # Test whether the ssc got from the fastcan is consistent
    # with the ssc got from CCA
    n_samples = 200
    n_features = 20
    n_targets = 10
    n_informative = 10

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        noise=0.1,
        shuffle=False,
        random_state=0,
    )

    cca = CCA(n_components=n_targets)
    cca.fit(X, y)
    X_c, Y_c = cca.transform(X, y)
    corrcoef = np.corrcoef(X_c, Y_c, rowvar=False).diagonal(offset=n_targets)
    gtruth_ssc = sum(corrcoef**2)

    correlation_filter = FastCan(
        n_features_to_select=n_features,
    )
    correlation_filter.fit(X, y)
    ssc = correlation_filter.scores_.sum()
    assert_almost_equal(actual=ssc, desired=gtruth_ssc)

def test_h_eta_consistency():
    # Test whether the ssc got from h-correlation is
    # consistent with the ssc got from eta-cosine
    n_samples = 200
    n_features = 20
    n_targets = 10
    n_informative = 10
    n_to_select = 5

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        noise=0.1,
        shuffle=False,
        random_state=0,
    )

    h_correlation = FastCan(
        n_features_to_select=n_to_select,
        eta=False
    )
    eta_cosine = FastCan(
        n_features_to_select=n_to_select,
        eta=True
    )
    h_correlation.fit(X, y)
    eta_cosine.fit(X, y)
    assert_array_almost_equal(h_correlation.scores_, eta_cosine.scores_)

def test_raise_errors():
    # Test whether fastcan raise errors properly
    n_samples = 20
    n_features = 20
    n_classes = 10
    n_informative = 15
    n_redundant = n_features-n_informative

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        n_redundant=n_redundant,
        flip_y=0,
        shuffle=False,
        random_state=0,
    )

    selector_n_select = FastCan(
        n_features_to_select=n_features+1,
    )
    selector_n_inclusive = FastCan(
        n_features_to_select=n_features,
        inclusive_indices=range(n_features+1)
    )
    selector_eta_for_small_size_samples = FastCan(
        n_features_to_select=n_features,
        eta=True
    )

    with pytest.raises(ValueError, match=r"n_features_to_select .*"):
        selector_n_select.fit(X, y)

    with pytest.raises(ValueError, match=r"n_inclusions .*"):
        selector_n_inclusive.fit(X, y)

    with pytest.raises(ValueError, match=r"`eta` cannot be True, .*"):
        selector_eta_for_small_size_samples.fit(X, y)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_cython_errors():
    # Test whether fastcan raise cython errors properly
    rng = np.random.default_rng(0)
    n_samples = 20
    n_informative = 3
    x_sub = rng.random((n_samples, n_informative))
    y = rng.random((n_samples))


    selector_zero_vector = FastCan(
        n_features_to_select=n_informative+1,
    )

    with pytest.raises(
        ZeroDivisionError,
        match="Cannot normalize a vector of all zeros."
    ):
        # Zeros vector during orthogonalization
        selector_zero_vector.fit(np.c_[x_sub, x_sub[:, 0]], y)

    with pytest.raises(
        ZeroDivisionError,
        match="Cannot normalize a matrix containing a vector of all zeros."
    ):
        # Constant vector
        selector_const_vector = FastCan(
            n_features_to_select=2,
        )
        selector_const_vector.fit(np.zeros((3, 2)), [1, 2, 3])

    with pytest.raises(RuntimeError, match=r"No candidate feature can .*"):
        # No candidate
        selector_zero_vector.fit(np.c_[x_sub, x_sub[:, 0]+x_sub[:, 1]], y)

