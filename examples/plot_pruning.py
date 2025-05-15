"""
============
Data pruning
============

.. currentmodule:: fastcan

This example shows how to prune dataset with :func:`minibatch`
based on :class:`FastCan`.
The method is compared to random data pruning.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

# %%
# Load data and prepare baseline
# ------------------------------
# We use ``iris`` dataset and logistic regression model to demonstrate data pruning.
# The baseline model is a logistic regression model trained on the entire dataset.
# The coefficients of the model trained on the pruned dataset will be compared to
# the baseline model with R-squared score.
# The higher R-squared score, the better the pruning.

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

data, labels = load_iris(return_X_y=True)
baseline_lr = LogisticRegression(max_iter=1000).fit(data, labels)

# %%
# Random data pruning
# -------------------
# There are 150 samples in the dataset. The pruned dataset for
# random pruning method is selected randomly.

import numpy as np


def _random_pruning(X, y, n_samples_to_select: int, random_state: int):
    rng = np.random.default_rng(random_state)
    ids_random = rng.choice(y.size, n_samples_to_select, replace=False)
    pruned_lr = LogisticRegression(max_iter=1000).fit(X[ids_random], y[ids_random])
    return pruned_lr.coef_, pruned_lr.intercept_


# %%
# FastCan data pruning
# --------------------
# To use :class:`FastCan` to prune the data, there are two steps:
#
# #. Learn the atoms by Dictionary Learning (here we use ``KMeans``)
# #. Select the samples by :func:`minibatch` according to the multiple correlation
#    between each atom and the batch of samples.

from sklearn.cluster import KMeans

from fastcan import minibatch


def _fastcan_pruning(
    X,
    y,
    n_samples_to_select: int,
    random_state: int,
    n_atoms: int,
    batch_size: int,
):
    kmeans = KMeans(
        n_clusters=n_atoms,
        random_state=random_state,
    ).fit(X)
    atoms = kmeans.cluster_centers_
    ids_fastcan = minibatch(
        X.T, atoms.T, n_samples_to_select, batch_size=batch_size, verbose=0
    )
    pruned_lr = LogisticRegression(max_iter=1000).fit(X[ids_fastcan], y[ids_fastcan])
    return pruned_lr.coef_, pruned_lr.intercept_


# %%
# Compare pruning methods
# -----------------------
# 100 samples are selected from 150 original data with ``Random`` pruning and
# ``FastCan`` pruning. The results show that ``FastCan`` pruning gives a higher
# mean value of R-squared and a lower standard deviation.

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def plot_box(X, y, baseline, n_samples_to_select: int, n_random: int):
    r2_fastcan = np.zeros(n_random)
    r2_random = np.zeros(n_random)
    for i in range(n_random):
        coef, intercept = _fastcan_pruning(
            X, y, n_samples_to_select, i, n_atoms=50, batch_size=2
        )
        r2_fastcan[i] = r2_score(
            np.c_[coef, intercept], np.c_[baseline.coef_, baseline.intercept_]
        )

        coef, intercept = _random_pruning(X, y, n_samples_to_select, i)
        r2_random[i] = r2_score(
            np.c_[coef, intercept], np.c_[baseline.coef_, baseline.intercept_]
        )

    plt.boxplot(np.c_[r2_fastcan, r2_random])
    plt.ylabel("R2")
    plt.xticks(ticks=[1, 2], labels=["FastCan", "Random"])
    plt.show()


plot_box(data, labels, baseline_lr, n_samples_to_select=100, n_random=100)
