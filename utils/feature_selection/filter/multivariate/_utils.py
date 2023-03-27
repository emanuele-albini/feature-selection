from typing import Union, Optional
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm
from sklearn.utils.validation import check_is_fitted

from ..univariate import UnivariateFilterMethod
from ._cmi import conditional_mutual_info_score


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

    def get_relevance(self, X: np.ndarray, y: np.ndarray, i: int):
        relevance_func = get_function(self.relevance)
        kwargs = self.relevance_kwargs.copy() or {}

        if 'discrete_target' in kwargs:
            kwargs['y_discrete'] = kwargs.pop('discrete_target')
        if 'discrete_features' in kwargs:
            discrete = kwargs.pop('discrete_features')
            if not isinstance(discrete, bool):
                discrete = discrete[i]
            kwargs['x_discrete'] = discrete

        return relevance_func(X[:, i], y, **kwargs)

    def get_relevances(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       progress_bar: bool = False,
                       progress_bar_kwargs: Optional[dict] = None):
        relevances = np.full(X.shape[1], np.nan)
        for i in tqdm(range(X.shape[1]), disable=not progress_bar, **(progress_bar_kwargs or {})):
            relevances[i] = self.get_relevance(X, y, i)
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

    def get_redundancy(self, X: np.ndarray, i: int, j: int, cache: Optional[np.ndarray] = None):
        if cache is not None:
            # Use the cache if possible
            if np.isnan(cache[i, j]):
                return cache[i, j]
            # If the cache is symmetric, we can use it
            elif np.isnan(cache[j, i]) and self.symmetric_redundancy:
                return cache[j, i]
        else:
            assert cache.shape == (X.shape[1], X.shape[1]), "Cache must be a square (nb_features x nb_features) matrix."

        # Else, compute redundancy from scratch
        redundancy_func = get_function(self.redundancy)
        kwargs = self.redundancy_kwargs.copy() or {}

        # Handle discrete features
        if 'discrete_features' in kwargs:
            discrete = kwargs.pop('discrete_features')
            if not isinstance(discrete, bool):
                kwargs['x_discrete'] = discrete[i]
                kwargs['y_discrete'] = discrete[j]
            else:
                kwargs['x_discrete'] = kwargs['y_discrete'] = discrete

        if 'discrete_target' in kwargs:
            raise ValueError(
                "Redundancy functions are between features! They cannot accept a discrete_target as an argument.")

        # Compute redundancy
        redundancy = cache[i, j] = redundancy_func(X[:, i], X[:, j], **kwargs)

        return redundancy

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

    def get_conditional_relevance(self, X: np.ndarray, y: np.ndarray, i: int, j: int):
        conditional_relevance_func = get_function(self.conditional_relevance)
        kwargs = self.conditional_relevance_kwargs.copy() or {}

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

        return conditional_relevance_func(x=X[:, i], y=y, z=X[:, j], **kwargs)

    def get_conditional_relevance_name(self):
        # Let's get the conditional relevance function (if it fails we will get an error)
        _ = get_function(self.conditional_relevance)
        return get_function_name(self.conditional_relevance)


class RankingSelectorMixin(ABC):
    def __init__(self, *args, k: int = None, **kwargs):
        self.k = k
        super().__init__(*args, **kwargs)

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

        if k is None:
            raise ValueError('k must be passed as an argument to `get_support` or as an argument to the constructor.')

        # Check if the filter has been fitted
        check_is_fitted(self, 'ranking_')

        # We set the ranks of the features that are not selected to infinity
        ranking = self.ranking_
        ranking = np.nan_to_num(ranking, nan=np.inf)

        # We return the mask or the indices accordingly
        mask = ranking <= k
        return mask if not indices else np.where(mask)[0]