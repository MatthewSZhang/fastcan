.. currentmodule:: fastcan

.. _unsupervised:

==============================
Unsupervised feature selection
==============================

We can use :class:`FastCan` to do unsupervised feature selection.
The unsupervised application of :class:`FastCan` tries to select features, which
maximize the sum of the squared canonical correlation (SSC) with the principal
components (PCs) acquired from PCA (principal component analysis) of the feature
matrix :math:`X`.

    >>> from sklearn.decomposition import PCA
    >>> from sklearn import datasets
    >>> from fastcan import FastCan
    >>> iris = datasets.load_iris()
    >>> X = iris["data"]
    >>> y = iris["target"]
    >>> f_names = iris["feature_names"]
    >>> t_names = iris["target_names"]
    >>> pca = PCA(n_components=2)
    >>> X_pcs = pca.fit_transform(X)
    >>> selector = FastCan(n_features_to_select=2, verbose=0)
    >>> selector.fit(X, X_pcs[:, :2])
    >>> selector.indices_
    array([2, 1], dtype=int32)

.. note::
    There is no guarantee that this unsupervised :class:`FastCan` will select
    the optimal subset of the features, which has the highest SSC with PCs.
    Because :class:`FastCan` selects features in a greedy manner, which may lead to
    suboptimal results. See the following plots.

.. plot::
    :context: close-figs
    :align: center

    from itertools import combinations
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import CCA

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
        """
        n_components = min(X.shape[1], y.shape[1])
        cca = CCA(n_components=n_components)
        X_c, y_c = cca.fit_transform(X, y)
        corrcoef = np.diagonal(
            np.corrcoef(X_c, y_c, rowvar=False),
            offset=n_components
        )
        return sum(corrcoef**2)

    comb = list(combinations([0, 1, 2, 3], 2))
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(8, 6), layout="constrained")
    for i in range(2):
        for j in range(3):
            f1_idx = comb[i*3+j][0]
            f2_idx = comb[i*3+j][1]
            score = ssc(X[:, [f1_idx, f2_idx]], X_pcs)
            scatter = axs[i, j].scatter(X[:, f1_idx], X[:, f2_idx], c=y)
            axs[i, j].set(xlabel=f_names[f1_idx], ylabel=f_names[f2_idx])
            axs[i, j].set_title(f"SSC: {score:.3f}")
    for spine in axs[1, 0].spines.values():
            spine.set_edgecolor('red')
    _ = axs[1, 2].legend(scatter.legend_elements()[0], t_names, loc="lower right")

