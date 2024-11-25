"""Test feature selection extend"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import make_classification

from fastcan import FastCan, extend


def test_select_extend_cls():
    # Test whether refine work correctly with random samples.
    n_samples = 200
    n_features = 30
    n_informative = 20
    n_classes = 8
    n_repeated = 5
    n_to_select = 18

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )

    n_features_to_select = 2
    selector = FastCan(n_features_to_select).fit(X, y)
    indices = extend(selector, n_to_select, batch_size=3)
    selector_inc = FastCan(n_features_to_select, indices_include=[10]).fit(X, y)
    indices_inc = extend(selector_inc, n_to_select, batch_size=3)
    selector_exc = FastCan(
        n_features_to_select, indices_include=[10], indices_exclude=[0]
    ).fit(X, y)
    indices_exc = extend(selector_exc, n_to_select, batch_size=3)


    assert np.unique(indices).size == n_to_select
    assert_array_equal(indices[:n_features_to_select], selector.indices_)
    assert np.unique(indices_inc).size == n_to_select
    assert_array_equal(indices_inc[:n_features_to_select], selector_inc.indices_)
    assert np.unique(indices_exc).size == n_to_select
    assert_array_equal(indices_exc[:n_features_to_select], selector_exc.indices_)
    assert ~np.isin(0, indices_exc)


def test_extend_error():
    # Test refine raise error.
    n_samples = 200
    n_features = 20
    n_informative = 10
    n_classes = 8
    n_repeated = 5

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )

    n_features_to_select = 2

    selector = FastCan(n_features_to_select, indices_include=[0]).fit(X, y)

    with pytest.raises(ValueError, match=r"n_features_to_select .*"):
        _ = extend(selector, n_features+1, batch_size=3)

    with pytest.raises(ValueError, match=r"The number of features to select .*"):
        _ = extend(selector, n_features_to_select, batch_size=3)

    with pytest.raises(ValueError, match=r"The size of mini batch without .*"):
        _ = extend(selector, n_features, batch_size=1)
