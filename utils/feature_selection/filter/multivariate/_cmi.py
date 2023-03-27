"""
    Conditional Mutual Information (CMI) Estimators
    
    Credits:
    `_cmi_cc` and `_cmi_cd` functions are partially based on the pycit package (MIT License)
    See the orirginal code at https://github.com/syanga/pycit

    Note: This estimators must be throughout tested with well-known distributions.

"""
__author__ = "Emanuele Albini"
__credits__ = ["PyCit Contributors"]
__all__ = [
    "conditional_mutual_info_score",
    "contingency_hypercube",
]

from tqdm import tqdm
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from sparse import COO  # Extension of scipy.sparse for 2D+ sparse arrays
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state

from ....preprocessing import sample_data


def __contingency_elements(X):
    for x in X.T:
        uniques, unique_idx = np.unique(x, return_inverse=True)
        n_uniques = uniques.shape[0]
        yield uniques, unique_idx, n_uniques


def contingency_hypercube(X, *, eps=None, sparse=False, dtype=np.int64):
    """Build a contingency hypercube of the data.
    It will contain the number of occurrences of each combination of variables in the data.

    This is the n-dimensional generalization of `sklearn.metrics.cluster.contingency_matrix`.

    Parameters
    ----------
    X : int array, shape = [n_samples, n_features]
        Data (dicrete) on which to compute the contingency.
    eps : float, default=None
        If a float, that value is added to all values in the contingency
        hypercube. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.
    sparse : bool, default=False
        If `True`, return a sparse COO contingency hypercube. If `eps` is not
        `None` and `sparse` is `True` will raise ValueError.
        .. versionadded:: 0.18
    dtype : numeric type, default=np.int64
        Output dtype. Ignored if `eps` is not `None`.
        .. versionadded:: 0.24
    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_uniques_0, n_uniques_1, ..., n_uniques_{n_features-1}]
        Hypercube :math:`C` such that :math:`C_{i, j, k, ...}` is the number of samples with value
        :math:`i` for X[:, 0],  :math:`j` or X[:, 1], and so forth. 
        If ``eps is None``, the dtype of this array will be integer unless set
        otherwise with the ``dtype`` argument. If ``eps`` is given, the dtype will be float.
        Will be a ``sparse.COO`` if ``sparse=True``, a np.ndarray otherwise.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    _, unique_idx, n_uniques = zip(*__contingency_elements(X))

    # Generate the consitency hypercube
    # Note: duplicates are summed together automatically by COO
    contingency = COO(
        (
            np.ones(len(X), dtype=dtype),  # Values
            tuple(unique_idx),  # Indices
        ),
        shape=tuple(n_uniques),
    )

    # Convert to dense if necessary
    if not sparse:
        contingency = contingency.todense()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps

    return contingency


def _cmi(x, y, z):
    """
        Conditional Mutual Information naive estimator for discrete variables.
        This is the definition of CMI. See https://en.wikipedia.org/wiki/Conditional_mutual_information

        x: data with shape (num_samples,)
        y: data with shape (num_samples,)
        z: conditioning data with shape (num_samples,)

        Returns:
            Conditional Mutual Information estimate
    """

    assert x.ndim == 1 and y.ndim == 1 and z.ndim == 1, "All inputs must be 1D arrays, not supported yet."
    xzy = np.concatenate((x.reshape(-1, 1) if x.ndim == 1 else x, z.reshape(-1, 1) if z.ndim == 1 else z,
                          y.reshape(-1, 1) if y.ndim == 1 else y),
                         axis=1)

    # Generate the contingency hypercube (does not support non-1D data)
    contingency = contingency_hypercube(xzy, sparse=True)

    # Get the contingency counts for the non-zero elements
    nzx, nzy, nzz = np.nonzero(contingency)
    count_tot = contingency.sum()
    counts_xzy = contingency[nzx, nzy, nzz]
    count_x = contingency.sum(axis=(1, 2)).todense().take(nzx).astype(np.int64, copy=False)
    count_z = contingency.sum(axis=(0, 2)).todense().take(nzz).astype(np.int64, copy=False)
    count_y = contingency.sum(axis=(0, 1)).todense().take(nzy).astype(np.int64, copy=False)

    # Since MI <= min(H(X), H(Y)), any labelling with zero entropy, i.e. containing a single cluster, implies MI = 0
    # if pi.size == 1 or pj.size == 1:
    # return 0.0

    # Compute the conditional mutual information
    log_counts_xzy = np.log(counts_xzy)
    log_counts_x_times_z = np.log(np.multiply(count_x, count_z))
    log_counts_y_times_z = np.log(np.multiply(count_y, count_z))
    log_counts_z = np.log(count_z)

    # Computation of MI can be simplified to:
    mi = counts_xzy / count_tot * (log_counts_z + log_counts_xzy - log_counts_x_times_z - log_counts_y_times_z)
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)

    return np.clip(mi.sum(), 0.0, None)


def _cmi_cd(x, y, z, k=5):
    """
        Conditional Mutual Information Estimator for continuous/discrete mixtures.
            - RAVK-GKOV CMI Estimator for discrete-continuous mixtures.
                RAVK [1], GKOV [2], see also [3].

        x: data with shape (num_samples, x_dim) or (num_samples,)
        y: data with shape (num_samples, y_dim) or (num_samples,)
        z: conditioning data with shape (num_samples, z_dim) or (num_samples,)
        k: number of nearest neighbors for estimation


        [1] A. Rahimzamani, H. Asnani, P. Viswanath, and S. Kannan
            Estimators for Multivariate Information Measures in General Probability Spaces
            Advances in Neural Information Processing Systems, 2018, vol. 31.
            https://proceedings.neurips.cc/paper_files/paper/2018/hash/c5ab6cebaca97f7171139e4d414ff5a6-Abstract.html
        [2] W. Gao, S. Kannan, S. Oh, and P. Viswanath
            Estimating Mutual Information for Discrete-Continuous Mixtures
            Advances in Neural Information Processing Systems, 2017, vol. 30. 
            https://proceedings.neurips.cc/paper/2017/hash/ef72d53990bc4805684c9b61fa64a102-Abstract.html
        [3] O. C. Mesner and C. R. Shalizi
            Conditional Mutual Information Estimation for Mixed, Discrete and Continuous Data
            IEEE Transactions on Information Theory, vol. 67, no. 1, pp. 464–484, Jan. 2021, doi: 10.1109/TIT.2020.3024886.

    """
    xzy = np.concatenate((x.reshape(-1, 1) if x.ndim == 1 else x, z.reshape(-1, 1) if z.ndim == 1 else z,
                          y.reshape(-1, 1) if y.ndim == 1 else y),
                         axis=1)

    lookup = NearestNeighbors(metric='chebyshev')
    lookup.fit(xzy)

    radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]
    radius = np.nextafter(radius[:, -1], 0)

    # modification for discrete-continuous
    k_list = k * np.ones(radius.shape, dtype='i')
    where_zero = np.array(radius == 0.0, dtype='?')
    if np.any(where_zero > 0):
        matches = lookup.radius_neighbors(xzy[where_zero], radius=0.0, return_distance=False)
        k_list[where_zero] = np.array([i.size for i in matches])

    x_dim = x.shape[1] if x.ndim > 1 else 1
    z_dim = z.shape[1] if z.ndim > 1 else 1

    # compute entropies
    lookup.fit(xzy[:, :x_dim + z_dim])
    n_xz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(xzy[:, x_dim:])
    n_yz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(xzy[:, x_dim:x_dim + z_dim])
    n_z = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    return np.mean(digamma(k_list) + digamma(n_z + 1.) - digamma(n_xz + 1.) - digamma(n_yz + 1.))


def _cmi_cc(x, y, z, k=5):
    """
        Conditional Mutual Information Estimator: I(X;Y|Z) for continuous variables.
        Methods:
        - Frenzel-Pompe CMI Estimator (based on KSG MI Estimator) [1]
            
        See also    
        
        x: data with shape (num_samples, x_dim) or (num_samples,)
        y: data with shape (num_samples, y_dim) or (num_samples,)
        z: conditioning data with shape (num_samples, z_dim) or (num_samples,)
        k: number of nearest neighbors for estimation

        [1] Partial Mutual Information for Coupling Analysis of Multivariate Time Series
            Stefan Frenzel and Bernd Pompe. 2007. Physics Review Letters.
    """

    xzy = np.concatenate((x.reshape(-1, 1) if x.ndim == 1 else x, z.reshape(-1, 1) if z.ndim == 1 else z,
                          y.reshape(-1, 1) if y.ndim == 1 else y),
                         axis=1)

    lookup = NearestNeighbors(metric='chebyshev')
    lookup.fit(xzy)

    radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]
    radius = np.nextafter(radius[:, -1], 0)

    x_dim = x.shape[1] if x.ndim > 1 else 1
    z_dim = z.shape[1] if z.ndim > 1 else 1

    # compute entropies
    lookup.fit(xzy[:, :x_dim + z_dim])
    n_xz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(xzy[:, x_dim:])
    n_yz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(xzy[:, x_dim:x_dim + z_dim])
    n_z = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    return digamma(k) + np.mean(digamma(n_z + 1.) - digamma(n_xz + 1.) - digamma(n_yz + 1.))


def _scale(x, random_state):
    # Get the random generator
    rng = check_random_state(random_state)

    # Scale
    x = scale(x, with_mean=False, copy=True).astype(np.float64)
    mean = np.maximum(1, np.mean(np.abs(x)))

    # Add small noise to continuous features as advised in Kraskov et. al.
    x += 1e-10 * mean * rng.standard_normal(size=len(x))
    return x


def _compute_cmi(x, y, z, x_discrete, y_discrete, z_discrete, n_neighbors, random_state):
    """Compute mutual information between two variables.
    This is a simple wrapper which selects a proper function to call based on
    whether `x` and `y` are discrete or not.
    """
    discrete = np.array([x_discrete, y_discrete, z_discrete])

    if not x_discrete:
        x = _scale(x, random_state=random_state)
    if not y_discrete:
        y = _scale(y, random_state=random_state)
    if not z_discrete:
        z = _scale(z, random_state=random_state)

    if np.all(discrete):
        return _cmi(x, y, z)
    elif not np.any(discrete):
        return _cmi_cc(x, y, z, n_neighbors)
    else:
        return _cmi_cd(x, y, z, n_neighbors)


def conditional_mutual_info_score(
    x,
    y,
    z,
    random_state=0,
    n=np.inf,
    n_neighbors=3,
    **kwargs,
):

    # Preprocess data
    x, y, z = np.asarray(x).flatten(), np.asarray(y).flatten(), np.asarray(z).flatten()
    assert len(x) == len(y) == len(z), "X, y, and z must have the same length."
    assert not np.any(np.isnan(y)), "y must not contain NaNs."

    # Mask the NaN data
    mask = (~np.isnan(x)) & (~np.isnan(z))
    x, y, z = x[mask], y[mask], z[mask]

    # Sample data, it is crucial to use the same random state for all variables
    # (so that the connections between them are preserved)
    x = sample_data(x, n=n, random_state=random_state, replace=False, safe=True)
    y = sample_data(y, n=n, random_state=random_state, replace=False, safe=True)
    z = sample_data(z, n=n, random_state=random_state, replace=False, safe=True)

    # Compute scores for each feature
    return _compute_cmi(
        x,
        y,
        z,
        n_neighbors=n_neighbors,
        random_state=random_state,
        **kwargs,
    )
