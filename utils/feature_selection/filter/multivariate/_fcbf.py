"""This module implements the FCBF (Fast Correlation Based Filter) algorithm for feature selection.

See the following papers for more details:
[1] Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution.
Jieping Ye, and Huan Liu. IEEE Transactions on Knowledge and Data Engineering. 2003.
[2] Fast Correlation-Based Filter (FCBF) with a different search strategy.
Baris Senliol et al. 2008.

"""
__all__ = ["FCBF"]

from typing import Union, Optional
import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator

from ._utils import RankingSelectorMixin, RelevanceMixin, RedundancyMixin


class FCBF(RankingSelectorMixin, RelevanceMixin, RedundancyMixin, BaseEstimator):
    def __init__(
        self,
        relevance: Union[callable, str],
        redundancy: Union[callable, str],
        relevance_kwargs: Optional[dict] = None,
        redundancy_kwargs: Optional[dict] = None,
        symmetric_redundancy: bool = False,
        k: int = None,
        delta: float = 0.0,
        gamma: float = 1.0,
    ):
        """Initialize the FCBF multivariate filter.

        Args:
            relevance (Union[callable, str]): A function that takes two 1D arrays x and y and returns a relevance score.
            redundancy (Union[callable, str]): A function that takes two 1D arrays x and y and returns a redundancy score.
            relevance_kwargs (Optional[dict], optional): Additional arguments for the relevance function. Defaults to None.
            redundancy_kwargs (Optional[dict], optional): Additional arguments for the redundancy function. Defaults to None.
            symmetric_redundancy (bool, optional): If True, the algorithm will assume symmetric redundancy (less expensive computationally). Defaults to False.
            k (int, optional): Number of features to select. If None, it will run the algorithm on all features (and k must be passed to `get_support`). Defaults to None.
            delta (float, optional): The minimum relevance for the selected features. Defaults to 0.0.
                The higher the value, the larger the number of non-relevant features ignored will be.
            gamma (float, optional): The redundancy/relevance ratio below which a feature is considered redundant. Defaults to 1.0.
                The higher the value, the more redundant features will be selected.
        """

        self.delta = delta
        self.gamma = gamma
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
            FCBF : Fitted filter.
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
            progress_bar_kwargs=dict(desc=f'FCBF: Relevance ({self.get_relevance_name()})'),
            n_jobs=n_jobs,
        )

        # Let's setup a cache for the redundancy calculation
        if not hasattr(self, "_redundancy_cache"):
            self._redundancy_cache = np.full((n_features, n_features), np.nan)

        # Set of selected features (highest relevance first)
        selected_features = np.argsort(relevance)[::-1]

        n_iterations = self.get_n_iterations(n_features)
        for i in range(n_iterations):
            # Early stopping if there are no more features to select
            if i >= len(selected_features):
                break

            # Get the current feature
            current_feature = selected_features[i]
            current_relevance = relevance[current_feature]

            # Early stopping if the relevance is below the threshold
            if current_relevance <= self.delta:
                selected_features = selected_features[:i]
                break

            following_features = selected_features[i + 1:].copy()

            redundancies = self.get_redundancies(
                X,
                [(current_feature, j) for j in following_features],
                cache=self._redundancy_cache,
                progress_bar=progress_bar,
                progress_bar_kwargs=dict(desc=f'FCBF: Redundancy ({self.get_redundancy_name()}) {i}/{n_iterations}'),
                n_jobs=n_jobs,
            )

            for following_feature in following_features:
                # Get the redundancy
                redundancy_value = redundancies[(current_feature, following_feature)]
                relevance_value = relevance[following_feature]

                # Remove the feature from the selected features
                # if the y are more redundant (with already included features) than relevant
                if redundancy_value > self.gamma * relevance_value:
                    # NOTE: We have to use np.where because following_feature may change position during the loop
                    selected_features = np.delete(selected_features, np.where(selected_features == following_feature))

        ranking = np.full(n_features, np.nan)
        ranking[selected_features] = np.arange(len(selected_features)) + 1  # +1 because ranking starts at 1

        self.relevance_ = relevance
        self.ranking_ = ranking

        return self
