"""
Feature selection
"""

from numbers import Integral, Real

import numpy as np
from scipy.linalg import orth
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from ._cancorr_fast import _forward_search  # type: ignore


class FastCan(SelectorMixin, BaseEstimator):
    """Forward feature selector according to the sum of squared
    canonical correlation coefficients (SSC).


    Parameters
    ----------
    n_features_to_select : int, default=1
        The parameter is the absolute number of features to select.

    indices_include : array-like of shape (n_inclusions,), default=None
        The indices of the prerequisite features.

    eta : bool, default=False
        Whether to use eta-cosine method.

    tol : float, default=0.01
        Tolerance for linear dependence check.

        When abs(w.T*x) > `tol`, the modified Gram-Schmidt is failed as
        the feature `x` is linear dependent to the selected features,
        and `mask` for that feature will True.

    verbose : int, default=1
        The verbosity level.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    indices_ : ndarray of shape (n_features_to_select,), dtype=int
        The indices of the selected features. The order of the indices
        is corresponding to the feature selection process.

    support_ : ndarray of shape (n_features,), dtype=bool
        The mask of selected features.

    scores_ : ndarray of shape (n_features_to_select,), dtype=float
        The h-correlation/eta-cosine of selected features. The order of
        the scores is corresponding to the feature selection process.

    References
    ----------
    * Zhang, S., & Lang, Z. Q. (2022).
        Orthogonal least squares based fast feature selection for
        linear classification. Pattern Recognition, 123, 108419.

    * Zhang, S., Wang, T., Worden, K., Sun L., & Cross, E. J. (2025).
        Canonical-correlation-based fast feature selection for
        structural health monitoring. Mechanical Systems and Signal Processing,
        223, 111895.

    Examples
    --------
    >>> from fastcan import FastCan
    >>> X = [[1, 0], [0, 1]]
    >>> y = [1, 0]
    >>> FastCan(verbose=0).fit(X, y).get_support()
    array([ True, False])
    """

    _parameter_constraints: dict = {
        "n_features_to_select": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "indices_include": [None, "array-like"],
        "eta": ["boolean"],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_features_to_select=1,
        indices_include=None,
        eta=False,
        tol=0.01,
        verbose=1,
    ):
        self.n_features_to_select = n_features_to_select
        self.indices_include = indices_include
        self.eta = eta
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        """Preprare data for h-correlation or eta-cosine methods and select features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples, n_outputs)
            Target matrix.


        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_params()
        # X y
        check_X_params = {
            "ensure_min_samples": 2,
            "order": "F",
            "dtype": float,
            "force_writeable": True,
        }
        check_y_params = {
            "ensure_min_samples": 2,
            "ensure_2d": False,
            "order": "F",
            "dtype": float,
            "force_writeable": True,
        }
        X, y = self._validate_data(
            X=X,
            y=y,
            multi_output=True,
            validate_separately=(check_X_params, check_y_params),
        )
        check_consistent_length(X, y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = y.reshape(-1, 1)

        # indices_include
        if self.indices_include is None:
            indices_include = np.zeros(0, dtype=int)
        else:
            indices_include = check_array(
                self.indices_include,
                ensure_2d=False,
                dtype=int,
                ensure_min_samples=0,
            )

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]

        if self.n_features_to_select > n_features:
            raise ValueError(
                f"n_features_to_select {self.n_features_to_select} "
                f"must be <= n_features {n_features}."
            )

        if indices_include.ndim != 1:
            raise ValueError(
                f"Found indices_include with dim {indices_include.ndim}, "
                "but expected == 1."
            )

        if indices_include.size >= n_features:
            raise ValueError(
                f"n_inclusions {indices_include.size} must "
                f"be < n_features {n_features}."
            )

        if np.any((indices_include < 0) | (indices_include >= n_features)):
            raise ValueError(
                "Out of bounds. "
                f"All items in indices_include should be in [0, {n_features}). "
                f"But got indices_include = {indices_include}."
            )

        if (n_samples < n_features + n_outputs) and self.eta:
            raise ValueError(
                "`eta` cannot be True, when n_samples < n_features+n_outputs."
            )

        if self.eta:
            xy_hstack = np.hstack((X, y))
            xy_centered = xy_hstack - xy_hstack.mean(0)
            singular_values, unitary_arrays = np.linalg.svd(
                xy_centered, full_matrices=False
            )[1:]
            qxy_transformed = singular_values.reshape(-1, 1) * unitary_arrays
            qxy_transformed = np.asfortranarray(qxy_transformed)
            X_transformed = qxy_transformed[:, :n_features]
            y_transformed = orth(qxy_transformed[:, n_features:])
        else:
            X_transformed = X - X.mean(0)
            y_transformed = orth(y - y.mean(0))

        indices, scores = self._prepare_data(
            indices_include,
        )
        n_threads = _openmp_effective_n_threads()
        _forward_search(
            X=X_transformed,
            V=y_transformed,
            t=self.n_features_to_select,
            tol=self.tol,
            num_threads=n_threads,
            verbose=self.verbose,
            indices=indices,
            scores=scores,
        )
        support = np.zeros(shape=self.n_features_in_, dtype=bool)
        support[indices] = True
        self.indices_ = indices
        self.support_ = support
        self.scores_ = scores
        return self

    def _prepare_data(self, indices_include):
        """Prepare data for _forward_search()
        When h-correlation method is used, n_samples_ = n_samples.
        When eta-cosine method is used, n_samples_ = n_features+n_outputs.

        Parameters
        ----------
        indices_include : array-like of shape (n_inclusions,), dtype=int
        The indices of the prerequisite features.

        Returns
        -------
        mask : ndarray of shape (n_features,), dtype=np.ubyte, order='F'
            Mask for invalid candidate features.
            The data type is unsigned char.

        indices: ndarray of shape (n_features_to_select,), dtype=np.intc, order='F'
            The indices vector of selected features, initiated with -1.
            The data type is signed int.

        scores: ndarray of shape (n_features_to_select,), dtype=float, order='F'
            The h-correlation/eta-cosine of selected features.
        """
        # initiated with -1
        indices = np.full(self.n_features_to_select, -1, dtype=np.intc, order="F")
        indices[: indices_include.size] = indices_include
        scores = np.zeros(self.n_features_to_select, dtype=float, order="F")
        return indices, scores

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_
