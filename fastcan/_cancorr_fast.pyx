# Author: Matthew Sikai Zhang <matthew.szhang91@gmail.com>
#
# Fast feature selection with sum squared canoncial correlation coefficents
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cython cimport floating
from cython.parallel import prange
from scipy.linalg.cython_blas cimport isamax, idamax
from sklearn.utils._cython_blas cimport ColMajor, NoTrans
from sklearn.utils._cython_blas cimport _dot, _scal, _nrm2, _ger, _gemm, _axpy
from sklearn.utils._typedefs cimport int32_t, uint8_t

cdef int _iamax(
    int n, const floating *x,
    int incx,
) noexcept nogil:
    """
    Given a vector x, the function return the position of
    the vector element x(i) that has the largest absolute value for
    real flavors.
    """
    if floating is float:
        return isamax(&n, <float *> x, &incx) - 1
    else:
        return idamax(&n, <double *> x, &incx) - 1

cdef void _normv(
    floating[::1] x,            # IN/OUT
) except * nogil:
    """
    Vector normalization by Euclidean norm.
    x (IN) : (1, n_samples) Vector.
    x (OUT) : (1, n_samples) Normalized vector.
    """
    cdef:
        unsigned int n_samples = x.shape[0]
        floating x_norm

    x_norm = _nrm2(n_samples, &x[0], 1)
    if x_norm == 0.0:
        raise ZeroDivisionError("Cannot normalize a vector of all zeros.")
    x_norm = 1.0/x_norm
    _scal(n_samples, x_norm, &x[0], 1)

cdef void _normm(
    floating[::1, :] X,     # IN/OUT
) except * nogil:
    """
    Matrix column-wise normalization by Euclidean norm.
    X (IN) : (n_samples, nx) Matrix.
    X (OUT) : (n_samples, nx) Column-wise normalized matrix.
    """
    cdef:
        unsigned int n_samples = X.shape[0]
        unsigned int nx = X.shape[1]
        floating x_norm
        unsigned int j

    # X = X/norm(X)
    for j in range(nx):
        x_norm = _nrm2(n_samples, &X[0, j], 1)
        if x_norm == 0.0:
            raise ZeroDivisionError(
                "Cannot normalize a matrix containing a vector of all zeros."
            )
        x_norm = 1.0/x_norm
        _scal(n_samples, x_norm, &X[0, j], 1)


cdef floating _sscvm(
    const floating[::1] w,      # IN
    const floating[::1, :] V,   # IN
) noexcept nogil:
    """
    Sum of squared correlation coefficients.
    w : (n_samples,) Centred orthogonalized feature vector.
    V : (n_samples, nv) Centred orthogonalized target matrix.
    r2 : (nw, ) Sum of squared correlation coefficients, where r2i means the
        coefficient of determination between wi and V.
    """
    cdef:
        unsigned int n_samples = V.shape[0]
        unsigned int nv = V.shape[1]
        # R : (nw * nv) R**2 contains the pairwise h-correlation or eta-cosine, where
        #     rij means the h-correlation or eta-cosine between wi and vj.
        floating* r = <floating*> malloc(sizeof(floating) * nv)
        floating r2

    # r = w*V (w is treated as (1, n_samples))
    _gemm(ColMajor, NoTrans, NoTrans, 1, nv, n_samples, 1.0,
          &w[0], 1, &V[0, 0], n_samples, 0.0, r, 1)
    # r2 = r*r.T

    r2 = _dot(nv, r, 1, r, 1)

    free(r)
    return r2

cdef void _mgsvm(
    const floating[::1] w,      # IN
    floating[::1, :] X,         # IN/OUT
) noexcept nogil:
    """
    Modified Gram-Schmidt process. X = X - w.T*w*X
    w : (n_samples, ) Centred orthonormal selected feature vector.
    X (IN) : (n_samples, nx) Centred remaining feature matrix.
    X (OUT) : (n_samples, nx) Centred remaining feature matrix,
              which is orthogonal to w.
    """
    cdef:
        unsigned int n_samples = X.shape[0]
        unsigned int nx = X.shape[1]
        # r (1, nx)
        floating* r = <floating*> malloc(sizeof(floating) * nx)

    # r = w*X (w is treated as (1, n_samples))
    _gemm(ColMajor, NoTrans, NoTrans, 1, nx, n_samples, 1.0,
          &w[0], 1, &X[0, 0], n_samples, 0.0, r, 1)
    # X = X - w.T*r
    _ger(ColMajor, n_samples, nx, -1.0, &w[0], 1, r, 1, &X[0, 0], n_samples)

    free(r)

cdef void _mgsvv(
    const floating[::1] w,      # IN
    floating[::1] x,            # IN/OUT
) noexcept nogil:
    """
    Modified Gram-Schmidt process. x = x - w*w.T*x
    w : (n_samples, ) Centred orthonormal selected feature vector.
    x (IN) : (n_samples, ) Centred remaining feature vector.
    x (OUT) : (n_samples, ) Centred remaining feature vector, which is orthogonal to w.
    """
    cdef:
        unsigned int n_samples = x.shape[0]
        floating r

    # r = w.T*x
    r = _dot(n_samples, &w[0], 1, &x[0], 1)
    # x = x - w*r
    _axpy(n_samples, -r, &w[0], 1, &x[0], 1)


cdef void _orth(
    floating[::1, :] X,         # IN/OUT
) except * nogil:
    """Orthogonalization of a matrix by the modified Gram-Schmidt.
    X (IN) : (n_samples, n_features) Matrix.
    X (OUT) : (n_samples, n_features) Orthonormal matrix.

    Note: do not use scipy.linalg.orth which use Householder
    transformation. As classical/modified Gram-Schmidt orthognalize
    features in order, the corresponding scores reflect their
    importance, while Householder will mix the feature importance
    together.

    Parameters
    ----------
    n_features: integer greater or equal to 0
    """
    cdef:
        unsigned int n_features = X.shape[1]
        unsigned int i

    for i in range(n_features):
        if i == 0:
            _normv(X[:, 0])
        else:
            _mgsvm(X[:, i-1], X[:, i:])
            _normv(X[:, i])


cpdef void _forward_search(
    floating[::1, :] X,               # IN/OUT
    floating[::1, :] V,               # IN/OUT
    const unsigned int t,             # IN
    const floating tol,               # IN
    const unsigned int num_threads,   # IN
    const unsigned int verbose,       # IN
    uint8_t[::1] mask,                # IN/TEMP
    int32_t[::1] indices,             # OUT
    floating[::1] scores,             # OUT
) except * nogil:
    """
    Greedy search with SSC.
    X (IN) : (n_samples, n_features) Centered feature matrix.
    V (IN) : (n_samples, n_outputs) Centered target matrix.
    W (OUT) : (n_samples, n_features) Centered normalized feature matrix, which
              is orthonormal to selected features and M.
    V (OUT) : (n_samples, n_outputs) Centered orthonormal target matrix.
    t : Non-negative integer. The number of features to be selected.
    tol : Tolerance for linear dependence check.
    mask (n_features, ) Mask for valid candidate features.
    indices: (t, ) The indices vector of selected features, initiated with -1.
    scores: (t, ) The h-correlation/eta-cosine of selected features.
    """
    cdef:
        unsigned int n_samples = X.shape[0]
        unsigned int n_features = X.shape[1]
        floating* r2 = <floating*> malloc(sizeof(floating) * n_features)
        unsigned int n_masked           # The number of masked features
        floating g, ssc = 0.0
        unsigned int i, j
        int index = -1

    memset(&r2[0], 0, n_features * sizeof(floating))

    for i in range(t):
        if i == 0:
            # Preprocessing
            _orth(V)
            _normm(X)
        else:
            mask[index] = False
            r2[index] = 0
            # Make X orthogonal to X[:, indices[i-1]]
            n_masked = n_features
            for j in prange(n_features, nogil=True, schedule="static",
                            chunksize=1, num_threads=num_threads):
                if mask[j]:
                    _mgsvv(X[:, index], X[:, j])
                    _normv(X[:, j])
                    # Linear dependence check
                    g = _dot(n_samples, &X[0, index], 1, &X[0, j], 1)
                    if abs(g) > tol:
                        mask[j] = False
                        r2[j] = 0
                    else:
                        n_masked -= 1

            if n_masked == n_features:
                raise RuntimeError(
                    "No candidate feature can be found to form a non-singular "
                    f"matrix with the selected {i} features."
                )
        if indices[i] != -1:
            index = indices[i]
            scores[i] = _sscvm(X[:, index], V)
        else:
            # Score for X
            for j in range(n_features):
                if mask[j]:
                    r2[j] = _sscvm(X[:, j], V)

            # Find max scores and update indices, X, mask, and scores
            index = _iamax(n_features, r2, 1)
            indices[i] = index
            scores[i] = r2[index]

        ssc += scores[i]
        if verbose == 1:
            with gil:
                print(f"Progress: {i+1}/{t}, SSC: {ssc:.5f}", end="\r")

    free(r2)
