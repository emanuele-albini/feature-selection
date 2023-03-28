"""This module implements the mRMR algorithm for feature selection.

"""
__all__ = ["MRMR"]

import time
import itertools
from typing import Union, Optional
import multiprocessing

from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
from sklearn.base import BaseEstimator

from ._utils import RankingSelectorMixin, RelevanceMixin, RedundancyMixin


class MRMR(RankingSelectorMixin, RelevanceMixin, RedundancyMixin, BaseEstimator):
    def __init__(
        self,
        relevance: Union[callable, str],
        redundancy: Union[callable, str],
        relevance_kwargs: Optional[dict] = None,
        redundancy_kwargs: Optional[dict] = None,
        symmetric_redundancy: bool = False,
        k: int = None,
        quotient: bool = False,
    ):
        """Initialize the mRMR multivariate filter.

        Args:
            relevance (Union[callable, str]): A function that takes two 1D arrays x and y and returns a relevance score.
            redundancy (Union[callable, str]): A function that takes two 1D arrays x and y and returns a redundancy score.
            relevance_kwargs (Optional[dict], optional): Additional arguments for the relevance function. Defaults to None.
            redundancy_kwargs (Optional[dict], optional): Additional arguments for the redundancy function. Defaults to None.
            symmetric_redundancy (bool, optional): If True, the algorithm will assume symmetric redundancy (less expensive computationally). Defaults to False.
            k (int, optional): Number of features to select. If None, it will run the algorithm on all features (and k must be passed to `get_support`). Defaults to None.
            quotient (bool, optional): If True, it will use the quotient instead of the difference of relevance and reducndancy. Defaults to False.

        Note that this class only implements the "metahuristic" algorithm part of the algorithm.
        It does not directly implement the relevance and redundancy functions.
        

        See the following papers for more details on the algorithm:
        [1] Minimum reducndancy feature selection from microarray gene expression data.
            Chris Ding, Hanchuan Peng. Bioinformatics. 2005.
        [2] Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy.
            Hanchuan Peng, Fuhui Long, and Chris Ding. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2005.

        """

        self.quotient = quotient

        super().__init__(
            k=k,
            relevance=relevance,
            redundancy=redundancy,
            relevance_kwargs=relevance_kwargs,
            redundancy_kwargs=redundancy_kwargs,
            symmetric_redundancy=symmetric_redundancy,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        progress_bar: bool = False,
        n_jobs: int = 1,
    ):
        """Fit the feature selection filter based on the mRMR algorithm.

        Args:
            X (np.ndarray): Training data (features)
            y (np.ndarray): Target data (labels)
            progress_bar (bool, optional): If True, it will show progress bars during computation. Defaults to False.

        Returns:
            MRMR : Fitted filter.
        """

        # Preprocess data
        X, y = np.asarray(X), np.asarray(y).flatten()
        n_features = X.shape[1]

        # Let's setup the result arrays
        # - Relevance with the target
        # - Redundancy with the selected features
        # - Total score
        relevance = self.get_relevances(
            X,
            y,
            progress_bar=progress_bar,
            progress_bar_kwargs=dict(desc=f'mRMR: Relevance ({self.get_relevance_name()})'),
            n_jobs=n_jobs,
        )
        redundancy = np.full(n_features, np.nan)
        score = np.full(n_features, np.nan)

        # Let's setup a cache for the redundancy calculation
        redundancy_cache = np.full((n_features, n_features), np.nan)

        # Initialize selected features
        ranking = np.full(n_features, np.nan)

        n_iterations = self.get_n_iterations(n_features)
        for iteration in range(n_iterations):
            selected_features = np.where(~np.isnan(ranking))[0]
            remaining_features = np.where(np.isnan(ranking))[0]

            # Weight the redundancy by the number of selected features
            if iteration == 0:
                redundancy_ = np.ones(n_features)
            else:
                # Compute the redundancy of feature i with the selected features
                ijs = itertools.product(remaining_features, selected_features)
                redundancies = self.get_redundancies(
                    X,
                    ijs,
                    cache=redundancy_cache,
                    progress_bar=progress_bar,
                    n_jobs=n_jobs,
                    progress_bar_kwargs=dict(
                        desc=f'mRMR: Redundancy ({self.get_redundancy_name()}) ({iteration + 1}/{n_iterations})'),
                )
                redundancy_ = np.nan_to_num(redundancies, nan=0.0).sum(axis=1) / len(selected_features)

            # Avoid division by zero (when quotient=True)
            if self.quotient:
                redundancy_ = np.maximum(redundancy_, 1e-10)

            scores_ = relevance / redundancy_ if self.quotient else relevance - redundancy_

            # Select the best feature
            best_feature = np.argmax(np.where(np.isnan(ranking), scores_, -np.inf))

            # Update results
            score[best_feature] = scores_[best_feature]
            redundancy[best_feature] = redundancy_[best_feature]
            ranking[best_feature] = iteration + 1  # 1-indexed

        self.ranking_ = ranking
        self.relevance_ = relevance
        self.redundancy_ = redundancy

        return self