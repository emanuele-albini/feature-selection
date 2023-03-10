__author__ = "Emanuele Albini"
__all__ = [
    'UnivariateFilterMethod',
    'get_filter_scoring_function',
]

from enum import Enum

import numpy as np
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from scipy.stats import spearmanr, pearsonr, kruskal
from tqdm import tqdm

from ._relief import reliefF
from ....preprocessing import sample_data


class UnivariateFilterMethod(Enum):
    # Statistical tests
    CHI2 = 'chi2'
    FISHER_F_TEST = 'fisher'
    KRUSKAL_WALLIS_H_TEST = 'kruskal'
    SPEARMAN = 'spearman'
    PEARSON = 'pearson'
    # Information theory
    MUTUAL_INFO = 'mi'
    # RELIEF
    RELIEF_F = 'relief-f'


def _clean_inputs(f):
    def score_func(X, y):
        X, y = np.asarray(X), np.asarray(y).flatten()
        return f(X, y)

    return score_func


def _scipy_statistics(f, abs=False, negative=False, desc=None):
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

    def score_func(X, y):
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

    return score_func


def _sample(f, n=None, random_state=0):
    def score_func(X, y):
        X = sample_data(X, n=n, random_state=random_state, replace=False, safe=True)
        y = sample_data(y, n=n, random_state=random_state, replace=False, safe=True)
        return f(X, y)

    return score_func


def _univariate_without_nan_decorator(f, desc=None):
    def score_func(X, y):
        X, y = np.asarray(X), np.asarray(y).flatten()

        Mask = ~np.isnan(X)

        scores = []
        pvalues = []

        for i in tqdm(range(X.shape[1]), desc=desc, disable=desc is None):
            scores_ = f(X[:, i][Mask[:, i]].reshape(-1, 1), y[Mask[:, i]])
            if isinstance(scores_, tuple):
                scores_, pvalues_ = scores_
            else:
                pvalues_ = None

            scores.append(scores_[0])
            if pvalues_ is not None:
                pvalues.append(pvalues_[0])

        if pvalues:
            return np.array(scores), np.array(pvalues)
        else:
            return np.array(scores)

    return score_func


def get_filter_scoring_function(method, **kwargs):
    # Transform string to enum
    method = UnivariateFilterMethod(method)

    if len(kwargs) > 0 and (method not in [UnivariateFilterMethod.RELIEF_F, UnivariateFilterMethod.MUTUAL_INFO]):
        raise ValueError(f'No kwargs allowed for method {method}')

    # Filter methods
    if method == UnivariateFilterMethod.CHI2:
        score_func = _univariate_without_nan_decorator(chi2, desc=method.value)
    elif method == UnivariateFilterMethod.FISHER_F_TEST:
        score_func = _univariate_without_nan_decorator(f_classif, desc=method.value)
    elif method == UnivariateFilterMethod.KRUSKAL_WALLIS_H_TEST:
        # Statistic = 0.0 means identical distribution
        score_func = _univariate_without_nan_decorator(_scipy_statistics(kruskal, negative=True), desc=method.value)
    elif method == UnivariateFilterMethod.SPEARMAN:
        score_func = _univariate_without_nan_decorator(_scipy_statistics(spearmanr, abs=True), desc=method.value)
    elif method == UnivariateFilterMethod.PEARSON:
        score_func = _univariate_without_nan_decorator(_scipy_statistics(pearsonr, abs=True), desc=method.value)
    elif method == UnivariateFilterMethod.MUTUAL_INFO:
        score_func = _univariate_without_nan_decorator(_sample(mutual_info_classif, **kwargs), desc=method.value)
    elif method == UnivariateFilterMethod.RELIEF_F:
        score_func = lambda X, y: reliefF(X, y, **kwargs)
    else:
        raise ValueError('Unknown method: {}'.format(method))

    score_func = _clean_inputs(score_func)

    return score_func
