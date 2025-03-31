"""Test Jacobian matrix of NARX"""

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from fastcan.narx import NARX


def test_simple():
    """Simple model
    test model: y(k) = 0.4*y(k-1) + u(k-1) + 1
    initial y[-1] = 0
    u[0] = u[1] = u[2] = 1.5
    """
    # Ground truth
    X = np.array([1.5, 1.5, 1.5]).reshape(-1, 1)
    y = np.array([1, 2.9, 3.66]).reshape(-1, 1)

    feat_ids = np.array([1, 0]).reshape(-1, 1)
    delay_ids = np.array([1, 1]).reshape(-1, 1)
    output_ids = np.array([0, 0])
    coef = np.array([0.4, 1])
    intercept = np.array([1], dtype=float)
    sample_weight = np.array([1, 1, 1], dtype=float)


    y_hat = NARX._predict(
        NARX._expression,
        X=X,
        y_ref=y,
        coef=coef,
        intercept=intercept,
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
    )

    assert_array_equal(y_hat, y)

    delta_w = 0.00001
    coef_1 = np.array([0.4+delta_w, 1])

    y_hat_1 = NARX._predict(
        NARX._expression,
        X=X,
        y_ref=y,
        coef=coef_1,
        intercept=intercept,
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
    )


    e1 = y_hat_1 - y
    grad_truth = np.array([
        np.sum(e1*np.array([0, y_hat_1[0, 0], y_hat_1[1, 0]+coef_1[0]]).reshape(-1, 1)),
        np.sum(e1*np.array([0, X[0, 0], X[0, 0]*coef_1[0]+X[0, 0]]).reshape(-1, 1)),
        np.sum(e1*np.array([0, 1, coef_1[0]+1]).reshape(-1, 1)),
    ])


    cfd_ids = NARX._get_cfd_ids(feat_ids, delay_ids, output_ids, 1)
    grad = NARX._grad(
        np.r_[coef_1, intercept],
        NARX._expression,
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        sample_weight_sqrt=np.sqrt(sample_weight),
        cfd_ids=cfd_ids,
    )

    assert_almost_equal(grad.sum(axis=0), grad_truth, decimal=4)


def test_complex():
    """Complex model"""
    # Simulated model
    rng = np.random.default_rng(12345)
    n_samples = 2000
    max_delay = 3
    e0 = rng.normal(0, 0.1, n_samples)
    e1 = rng.normal(0, 0.02, n_samples)
    u0 = rng.uniform(0, 1, n_samples + max_delay)
    u1 = rng.normal(0, 0.1, n_samples + max_delay)
    y0 = np.zeros(n_samples + max_delay)
    y1 = np.zeros(n_samples + max_delay)
    for i in range(max_delay, n_samples + max_delay):
        y0[i] = (
            0.5 * y0[i - 1]
            + 0.8 * y1[i - 1]
            + 0.3 * u0[i] ** 2
            + 2 * u0[i - 1] * u0[i - 3]
            + 1.5 * u0[i - 2] * u1[i - 3]
            + 1
        )
        y1[i] = (
            0.6 * y1[i - 1]
            - 0.2 * y0[i - 1] * y1[i - 2]
            + 0.3 * u1[i] ** 2
            + 1.5 * u1[i - 2] * u0[i - 3]
            + 0.5
        )
    y = np.c_[y0[max_delay:] + e0, y1[max_delay:] + e1]
    X = np.c_[u0[max_delay:], u1[max_delay:]]

    feat_ids = np.array(
        [
            [-1, 2],
            [-1, 3],
            [0, 0],
            [0, 0],
            [0, 1],
            [-1, 3],
            [2, 3],
            [1, 1],
            [1, 0],
        ]
    )

    delay_ids = np.array(
        [
            [-1, 1],
            [-1, 1],
            [0, 0],
            [1, 3],
            [2, 3],
            [-1, 1],
            [1, 2],
            [0, 0],
            [2, 3],
        ]
    )

    output_ids = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

    coef = np.array(
        [
            0.5,
            0.8,
            0.3,
            2,
            1.5,
            0.6,
            -0.2,
            0.3,
            1.5,
        ]
    )

    intercept = np.array([1, 0.5])

    # NARX Jacobian
    cfd_ids = NARX._get_cfd_ids(feat_ids, delay_ids, output_ids, X.shape[1])
    grad = NARX._grad(
        np.r_[coef, intercept],
        NARX._expression,
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        sample_weight_sqrt=np.sqrt(np.ones((y.shape[0], 1))),
        cfd_ids=cfd_ids,
    )

    # Numerical gradient
    y_hat_0 = NARX._predict(
        NARX._expression,
        X=X,
        y_ref=y,
        coef=coef,
        intercept=intercept,
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
    )
    loss_0 = 0.5*np.sum((y_hat_0 - y)**2)

    delta_w = 0.00001
    for i in range(len(coef)+len(intercept)):
        if i < len(coef):
            coef_1 = np.copy(coef)
            coef_1[i] += delta_w
            intercept_1 = np.copy(intercept)
        else:
            coef_1 = np.copy(coef)
            intercept_1 = np.copy(intercept)
            intercept_1[i-len(coef)] += delta_w

        y_hat_1 = NARX._predict(
            NARX._expression,
            X=X,
            y_ref=y,
            coef=coef_1,
            intercept=intercept_1,
            feat_ids=feat_ids,
            delay_ids=delay_ids,
            output_ids=output_ids,
        )

        e1 = y_hat_1 - y
        loss_1 = 0.5*np.sum((e1)**2)
        grad_num = (loss_1 - loss_0) / delta_w

        assert_allclose(grad.sum(axis=0)[i], grad_num, rtol=1e-1)
