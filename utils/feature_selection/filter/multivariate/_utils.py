import os
import logging
from typing import Union, Optional
from abc import ABC, abstractmethod
import multiprocessing

from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..univariate import UnivariateFilterMethod
from ._cmi import conditional_mutual_info_score


def _get_joblib_temp_folder():
    if os.path.exists('/sys/fs/cgroup'):
        return '/sys/fs/cgroup'
    return None


def __remove_pvalues_decorator(function: callable):
    def wrapper(*args, **kwargs):
        r = function(*args, **kwargs)

        # Unpack scores and p-values (if any)
        if isinstance(r, tuple):
            scores, _ = r
        else:
            scores = r
            _ = None

        return scores

    return wrapper


def get_function(function: Union[callable, str]):
    if callable(function):
        return function
    elif isinstance(function, str):
        if function == 'cmi':
            return conditional_mutual_info_score
        else:
            # We get the univariate filter
            filter = UnivariateFilterMethod(function)

            # We ignore p-values
            function = __remove_pvalues_decorator(filter.scoring_function)

            # Use it as a scoring function between two arrays
            if filter.require_types:
                return lambda x, y, x_discrete=False, y_discrete=False, **kwargs: function(
                    np.expand_dims(x, axis=1), y, discrete_features=x_discrete, discrete_target=y_discrete, **kwargs)[0]
            else:
                return lambda x, y, **kwargs: function(np.expand_dims(x, axis=1), y, **kwargs)[0]
    else:
        raise ValueError(f"Invalid function: {function}. Must be a callable or a string.")


def get_function_name(function):
    if callable(function):
        return function.__name__
    else:
        return function


class RelevanceMixin(ABC):
    def __init__(
        self,
        *args,
        relevance: Union[callable, str],
        relevance_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.relevance = relevance
        self.relevance_kwargs = relevance_kwargs
        super().__init__(*args, **kwargs)

    def _get_relevance(self, x: np.ndarray, y: np.ndarray, i: int):
        relevance_func = get_function(self.relevance)
        kwargs = (self.relevance_kwargs or {}).copy()

        if 'discrete_target' in kwargs:
            kwargs['y_discrete'] = kwargs.pop('discrete_target')
        if 'discrete_features' in kwargs:
            discrete = kwargs.pop('discrete_features')
            if not isinstance(discrete, bool):
                discrete = discrete[i]
            kwargs['x_discrete'] = discrete

        return relevance_func(x, y, **kwargs)

    def get_relevances(
        self,
        X: np.ndarray,
        y: np.ndarray,
        progress_bar: bool = False,
        progress_bar_kwargs: Optional[dict] = None,
        n_jobs: int = 1,
    ):
        iters = tqdm(range(X.shape[1]), disable=not progress_bar, **(progress_bar_kwargs or {}))
        if n_jobs == 1:
            relevances = np.array([self._get_relevance(X[:, i], y, i) for i in iters])
        else:
            relevances = Parallel(n_jobs=min(multiprocessing.cpu_count() - 1, n_jobs),
                                  temp_folder=_get_joblib_temp_folder())(delayed(self._get_relevance)(X[:,
                                                                                                        i].copy(), y, i)
                                                                         for i in iters)
        return relevances

    def get_relevance_name(self):
        # Let's get the releance function (if it fails we will get an error)
        _ = get_function(self.relevance)
        return get_function_name(self.relevance)


class RedundancyMixin(ABC):
    def __init__(
        self,
        *args,
        redundancy: Union[callable, str],
        redundancy_kwargs: Optional[dict] = None,
        symmetric_redundancy: bool = False,
        **kwargs,
    ):
        self.redundancy = redundancy
        self.redundancy_kwargs = redundancy_kwargs
        self.symmetric_redundancy = symmetric_redundancy

        super().__init__(*args, **kwargs)

    def _get_redundancy(self,
                        x: np.ndarray,
                        y: np.ndarray,
                        i: int,
                        j: int,
                        ij_cache: float = None,
                        ji_cache: float = None):
        if ij_cache is not None:
            # Use the cache if possible
            if not np.isnan(ij_cache):
                return ij_cache
            # If the cache is symmetric, we can use it
            elif not np.isnan(ji_cache) and self.symmetric_redundancy:
                return ji_cache

        # Else, compute redundancy from scratch
        redundancy_func = get_function(self.redundancy)
        kwargs = (self.redundancy_kwargs or {}).copy()

        # Handle discrete features
        if 'discrete_features' in kwargs:
            discrete = kwargs.pop('discrete_features')
            if not isinstance(discrete, bool):
                kwargs['x_discrete'] = discrete[i]
                kwargs['y_discrete'] = discrete[j]
            else:
                kwargs['x_discrete'] = kwargs['y_discrete'] = discrete

        if 'discrete_target' in kwargs:
            logging.debug('Ignoring "discrete_target" argument. Redundancy is computed between two features.')
            del kwargs['discrete_target']

        return redundancy_func(x, y, **kwargs)

    def get_redundancy(self, X: np.ndarray, i: int, j: int, cache: Optional[np.ndarray] = None):
        if cache is not None:
            assert cache.shape == (X.shape[1], X.shape[1]), "Cache must be a square (nb_features x nb_features) matrix."
            cache_ij, cache_ji = cache[i, j], cache[j, i]
        else:
            cache_ij, cache_ji = None, None

        # Compute redundancy
        redundancy = cache[i, j] = self._get_redundancy(X[:, i], X[:, j], i, j, cache_ij, cache_ji)

        return redundancy

    def get_redundancies(
        self,
        X,
        ijs,
        cache: Optional[np.ndarray] = None,
        progress_bar: bool = False,
        progress_bar_kwargs: Optional[dict] = None,
        n_jobs: int = 1,
    ):
        if cache is not None:
            assert cache.shape == (X.shape[1], X.shape[1]), "Cache must be a square (nb_features x nb_features) matrix."

        ijs = list(ijs)
        iters = tqdm(ijs, disable=not progress_bar, **(progress_bar_kwargs or {}))

        # Compute redundancy
        if n_jobs == 1:

            redundancies = np.full((X.shape[1], X.shape[1]), np.nan)
            for i, j in iters:
                redundancies[i, j] = cache[i, j] = self._get_redundancy(
                    X[:, i],
                    X[:, j],
                    i,
                    j,
                    cache[i, j] if cache is not None else None,
                    cache[j, i] if cache is not None else None,
                )
        else:
            n_jobs = min(multiprocessing.cpu_count() - 1, n_jobs)
            X_sliced = [X[:, i].copy() for i in range(X.shape[1])]
            redundancies_ = Parallel(n_jobs=n_jobs,
                                     temp_folder=_get_joblib_temp_folder())(delayed(self._get_redundancy)(
                                         X_sliced[i],
                                         X_sliced[j],
                                         i,
                                         j,
                                         cache[i, j] if cache is not None else None,
                                         cache[j, i] if cache is not None else None,
                                     ) for i, j in iters)
            redundancies = np.full((X.shape[1], X.shape[1]), np.nan)
            for (i, j), redundancy in zip(ijs, redundancies_):
                redundancies[i, j] = cache[i, j] = redundancy

        return redundancies

    def get_redundancy_name(self):
        # Let's get the redundancy function (if it fails we will get an error)
        _ = get_function(self.redundancy)
        return get_function_name(self.redundancy)


class ConditionalRelevanceMixin(ABC):
    def __init__(self,
                 *args,
                 conditional_relevance: Union[callable, str],
                 conditional_relevance_kwargs: Optional[dict] = None,
                 **kwargs):
        self.conditional_relevance = conditional_relevance
        self.conditional_relevance_kwargs = conditional_relevance_kwargs
        super().__init__(*args, **kwargs)

    def _get_conditional_relevance(self, x, y, z, i, j):
        conditional_relevance_func = get_function(self.conditional_relevance)
        kwargs = (self.conditional_relevance_kwargs or {}).copy()

        if 'discrete_target' in kwargs:
            kwargs['y_discrete'] = kwargs.pop('discrete_target')

        if 'discrete_features' in kwargs:
            discrete = kwargs.pop('discrete_features')
            if not isinstance(discrete, bool):
                discrete_i = discrete[i]
                discrete_j = discrete[j]
            else:
                discrete_i = discrete_j = discrete

            kwargs['x_discrete'] = discrete_i
            kwargs['z_discrete'] = discrete_j

        return conditional_relevance_func(x=x, y=y, z=z, **kwargs)

    def get_conditional_relevance(self, X: np.ndarray, y: np.ndarray, i: int, j: int):
        return self._get_conditional_relevance(X[:, i], y, X[:, j], i, j)

    def get_conditional_relevances(
        self,
        X,
        y,
        ijs,
        progress_bar: bool = False,
        progress_bar_kwargs: Optional[dict] = None,
        n_jobs: int = 1,
    ):
        ijs = list(ijs)
        iters = tqdm(ijs, disable=not progress_bar, **(progress_bar_kwargs or {}))

        # Compute conditional relevance
        if n_jobs == 1:
            conditional_relevances = np.full((X.shape[1], X.shape[1]), np.nan)
            for i, j in iters:
                conditional_relevances[i, j] = self._get_conditional_relevance(
                    X[:, i],
                    y,
                    X[:, j],
                    i,
                    j,
                )
        else:

            X_sliced = [X[:, i].copy() for i in range(X.shape[1])]
            conditional_relevances_ = Parallel(n_jobs=n_jobs, temp_folder=_get_joblib_temp_folder())(
                delayed(self._get_conditional_relevance)(
                    X_sliced[i],
                    y,
                    X_sliced[j],
                    i,
                    j,
                ) for i, j in iters)
            conditional_relevances = np.full((X.shape[1], X.shape[1]), np.nan)
            for (i, j), conditional_relevance in zip(ijs, conditional_relevances_):
                conditional_relevances[i, j] = conditional_relevance

        return conditional_relevances

    def get_conditional_relevance_name(self):
        # Let's get the conditional relevance function (if it fails we will get an error)
        _ = get_function(self.conditional_relevance)
        return get_function_name(self.conditional_relevance)


class RankingSelectorMixin(ABC):
    def __init__(self, *args, k: int = None, **kwargs):
        self.k = k
        super().__init__(*args, **kwargs)

    def get_n_iterations(self, n_features: int = None):
        k = self.k or np.inf
        if n_features is not None:
            k = min(k, n_features)
        return k

    def get_support(self, k: int = None, indices: bool = False):
        """Get the support of the selected features.

        Args:
            k (int): Number of features to select
            indices (bool, optional): If True, it will return the indices of the selected features. Defaults to False.

        Returns:
            np.ndarray : A boolean mask or an array of indices
        """
        if k is None:
            k = self.k

        if k is None or np.isinf(k) or k < 0 or np.isnan(k):
            raise ValueError(
                'A finite k > 0 must be passed as an argument to `get_support` or as an argument to the constructor.')

        # Check if the filter has been fitted
        check_is_fitted(self, 'ranking_')

        # We set the ranks of the features that are not selected to infinity
        ranking = self.ranking_
        ranking = np.nan_to_num(ranking, nan=np.inf)

        # We return the mask or the indices accordingly
        mask = ranking <= k
        return mask if not indices else np.where(mask)[0]
