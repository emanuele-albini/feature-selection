__author__ = "Emanuele Albini"
__all__ = [
    'mutual_info',
    'entropy',
    'spearman',
    'pearson',
    'kruskal',
    'chi2',
    'fisher',
    'symmetric_uncertainty',
]

import functools

from tqdm import tqdm
import numpy as np
from scipy.stats import (
    spearmanr,
    pearsonr,
    kruskal as kruskalh,
    entropy as shannon_entropy,
    differential_entropy,
)
from sklearn.feature_selection import (
    chi2 as chi2_sklearn,
    f_classif as f_classif_sklearn,
    f_regression as f_regression_sklearn,
    mutual_info_classif as mutual_info_classif_sklearn,
    mutual_info_regression as mutual_info_regression_sklearn,
)
from sklearn.utils.validation import check_array

from ....preprocessing import sample_data


def _score_function_from_scipy_statistics_decorator_factory(abs: bool = False, negative: bool = False, desc=None):
    """Decorator that extracts the score and p-value from a scipy function.

    Args:
        f (callable): _description_
        abs (bool, optional): _description_. Defaults to False.
        negative (bool, optional): _description_. Defaults to False.
        desc (_type_, optional): _description_. Defaults to None.
    """
    def _extract_score(r):
        if hasattr(r, 'correlation'):
            return r.correlation
        elif hasattr(r, 'statistic'):
            return r.statistic
        elif isinstance(r, tuple):
            return r[0]
        else:
            raise NotImplementedError(f'Unsupported scipy function: {f}')

    def _extract_pvalue(r):
        if hasattr(r, 'pvalue'):
            return r.pvalue
        elif isinstance(r, tuple):
            return r[1]
        else:
            raise NotImplementedError(f'Unsupported scipy function: {f}')

    def decorator(f):
        @functools.wraps(f)
        def wrapper(X, y):
            # We have to ignore NaNs
            result = [f(X[:, i], y) for i in tqdm(range(X.shape[1]), desc=desc, disable=desc is None)]

            # Let's extract correlations/statistics and p-values
            scores = np.array([_extract_score(r) for r in result])
            pvalues = np.array([_extract_pvalue(r) for r in result])

            # If results are NaN at this point it mean that it could not find any kind of correlation
            scores = np.nan_to_num(scores, nan=0.0)
            pvalues = np.nan_to_num(pvalues, nan=0.0)

            if abs:
                scores = np.abs(scores)

            if negative:
                scores = -scores

            return scores, pvalues

        return wrapper

    return decorator


def _filter_decorator(f):
    """Decorator for our filter functions.

    - Adds supports for NaNs
    - Adds support for progress bar
    - Adds support for sampling

    Args:
        f (callable): Filter function

    Returns:
        callable: Filter function
    """
    @functools.wraps(f)
    def wrapper(
        X,
        y,
        n=np.inf,
        random_state=0,
        progress_bar=False,
        progress_bar_kwargs=None,
        **kwargs,
    ):

        # Preprocess data
        X, y = np.asarray(X), np.asarray(y).flatten()
        assert len(X) == len(y), f'X and y must have the same length (X: {len(X)}, y: {len(y)})'
        assert X.ndim == 2, f'X must be a 2D array (X: {X.ndim}D)'
        assert y.ndim == 1, f'y must be a 1D array (y: {y.ndim}D)'
        assert n > 0, f'n must be greater than 0 (n: {n})'
        assert not np.any(np.isnan(y)), f'y must not contain NaNs'

        # Pre-compute NaN mask
        Mask = ~np.isnan(X)

        scores = []
        pvalues = []

        for i in tqdm(range(X.shape[1]), disable=not progress_bar, **progress_bar_kwargs or {}):

            # Remove NaNs
            X_ = X[:, i][Mask[:, i]].reshape(-1, 1)
            y_ = y[Mask[:, i]]

            # Sample data
            X_ = sample_data(X_, n=n, random_state=random_state, replace=False, safe=True)
            y_ = sample_data(y_, n=n, random_state=random_state, replace=False, safe=True)

            # Compute scores for each feature
            scores_ = f(X_, y_, **kwargs)

            # Extract scores and p-values
            if isinstance(scores_, tuple):
                scores_, pvalues_ = scores_
            else:
                pvalues_ = None

            # Append scores and p-values
            scores.append(scores_[0])
            if pvalues_ is not None:
                pvalues.append(pvalues_[0])

        # Return scores and p-values accordingly
        if pvalues:
            return np.array(scores), np.array(pvalues)
        else:
            return np.array(scores)

    return wrapper


def _entropy(x, discrete=False, **kwargs):
    assert len(x.shape) == 1
    if discrete:
        # Compute discrete probability distribution
        _, counts = np.unique(x, return_counts=True)
        pk = counts / counts.sum()
        return shannon_entropy(pk, **kwargs)
    else:
        return differential_entropy(x, **kwargs)


@_filter_decorator
def entropy(X, y, discrete_features=False, **kwargs):

    n_features = X.shape[1]

    # Handle discrete features
    if isinstance(discrete_features, bool):
        discrete_mask = np.empty(n_features, dtype=bool)
        discrete_mask.fill(discrete_features)
    else:
        discrete_features = check_array(discrete_features, ensure_2d=False)
        if discrete_features.dtype != "bool":
            discrete_mask = np.zeros(n_features, dtype=bool)
            discrete_mask[discrete_features] = True
        else:
            discrete_mask = discrete_features

    # Compute entropies
    e = [_entropy(X[:, i], discrete=discrete_mask[i], **kwargs) for i in range(n_features)]

    return np.array(e)


@_filter_decorator
def mutual_info(*args, discrete_target=True, discrete_features=False, **kwargs):
    if discrete_target:
        return mutual_info_classif_sklearn(*args, discrete_features=discrete_features, **kwargs)
    else:
        return mutual_info_regression_sklearn(*args, discrete_features=discrete_features, **kwargs)


@_filter_decorator
def fisher(*arg, discrete_target=True, **kwargs):
    if discrete_target:
        return f_classif_sklearn(*arg, **kwargs)
    else:
        return f_regression_sklearn(*arg, **kwargs)


@_filter_decorator
def chi2(*args, **kwargs):
    return chi2_sklearn(*args, **kwargs)


@_filter_decorator
@_score_function_from_scipy_statistics_decorator_factory(abs=True)
def spearman(*args, **kwargs):
    return spearmanr(*args, **kwargs)


@_filter_decorator
@_score_function_from_scipy_statistics_decorator_factory(abs=True)
def pearson(*args, **kwargs):
    return pearsonr(*args, **kwargs)


@_filter_decorator
@_score_function_from_scipy_statistics_decorator_factory(negative=True)
def kruskal(*args, **kwargs):
    return kruskalh(*args, **kwargs)


@_filter_decorator
def symmetric_uncertainty(
    X,
    y,
    discrete_features=False,
    discrete_target=False,
    **kwargs,
):

    entropies_X = entropy(X, y, discrete_features=discrete_features, **kwargs)
    entropy_y = _entropy(y, discrete_features=discrete_target**kwargs)
    mi = mutual_info(X, y, discrete_features=discrete_features, discrete_target=discrete_target, **kwargs)

    return 2 * mi / (entropies_X + entropy_y)
