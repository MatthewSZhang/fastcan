# pylint: skip-file
"""Test FastCan"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_equal,
)

from fastcan import FastCan


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
        n_features_to_select=5,
    )
    correlation_filter.fit(X, y)
    ssc = correlation_filter.scores_.sum()
    assert_almost_equal(actual=ssc, desired=gtruth_ssc)

    support = correlation_filter.get_support()
    gtruth = np.zeros(20)
    gtruth[:5] = 1
    assert_array_equal(support, gtruth)
