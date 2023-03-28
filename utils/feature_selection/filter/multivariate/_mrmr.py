"""This module implements the mRMR algorithm for feature selection.

"""
__all__ = ["MRMR"]

from typing import Union, Optional
import numpy as np
from tqdm import tqdm

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
        )
        redundancy = np.full(n_features, np.nan)
        score = np.full(n_features, np.nan)

        # Let's setup a cache for the redundancy calculation
        redundancy_cache = np.full((n_features, n_features), np.nan)

        # Initialize selected features
        ranking = np.full(n_features, np.nan)

        n_iterations = min(self.k, n_features) if self.k is not None else n_features
        for iteration in range(n_iterations):
            selected_features = np.where(~np.isnan(ranking))[0]
            remaining_features = np.where(np.isnan(ranking))[0]

            # Compute the redundancy of feature i with the selected features
            redundancy_ = np.zeros(n_features)
            for i in tqdm(
                remaining_features,
                disable=not progress_bar,
                desc=f'mRMR: Redundancy ({self.get_redundancy_name()}) ({iteration + 1}/{n_iterations})'
            ):
                for j in selected_features:
                    redundancy_[i] += self.get_redundancy(X, i, j, redundancy_cache)

            # Weight the redundancy by the number of selected features
            if iteration == 0:
                redundancy_ = 0
            else:
                redundancy_ = redundancy_ / len(selected_features)

            # Avoid division by zero (when quotient=True)
            if self.quotient:
                redundancy_ = np.maximum(redundancy_, 1e-10)

            scores_ = relevance / redundancy_ if self.quotient else relevance - redundancy_

            # Select the best feature
            best_feature = np.argmax(scores_)

            # Update results
            score[best_feature] = scores_[best_feature]
            redundancy[best_feature] = redundancy_[best_feature]
            ranking[best_feature] = iteration + 1  # 1-indexed

        self.ranking_ = ranking
        self.relevance_ = relevance
        self.redundancy_ = redundancy

        return self