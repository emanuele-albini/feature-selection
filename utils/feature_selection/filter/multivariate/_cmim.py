""" This module implements the CMIM (Conditional Mutual Information Maximization) 
    algorithm for feature selection.
"""
__all__ = ['CMIM']

from typing import Union, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator
from ._utils import RankingSelectorMixin, RelevanceMixin, ConditionalRelevanceMixin


class CMIM(RankingSelectorMixin, RelevanceMixin, ConditionalRelevanceMixin, BaseEstimator):
    def __init__(
        self,
        relevance: Union[callable, str] = 'mi',
        relevance_kwargs: Optional[dict] = None,
        conditional_relevance: Union[callable, str] = 'cmi',
        conditional_relevance_kwargs: Optional[dict] = None,
        k: int = None,
    ):
        """Initialize the CMIM multivariate filter [1].

        Args:
            relevance (Union[callable, str]): A function that takes two 1D arrays x and y and returns a relevance score. Defaults to 'mi'.
            relevance_kwargs (Optional[dict], optional): Additional arguments for the relevance function. Defaults to None.
            conditional_relevance (Union[callable, str]): A function that takes three 1D arrays x, y and z and returns a conditional relevance score of x and y given z. Defaults to 'cmi'.
            conditional_relevance_kwargs (Optional[dict], optional): Additional arguments for the conditional relevance function. Defaults to None.
            k (int, optional): Number of features to select. If None, it will run the algorithm on all features (and k must be passed to `get_support`). Defaults to None.

        Note that this class only implements the "metahuristic" algorithm part of the algorithm.
        It does not directly implement the relevance and redundancy functions.
        But, by default, it uses the mutual information and conditional mutual information functions
        as described in the orginal paper [1].

        [1] Fast Binary Feature Selection with Conditional Mutual Information
            Francois Fleuret. 2004. Journal of Machine Learning Research.

        """

        super().__init__(
            k=k,
            relevance=relevance,
            relevance_kwargs=relevance_kwargs,
            conditional_relevance=conditional_relevance,
            conditional_relevance_kwargs=conditional_relevance_kwargs,
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
            CMIM : Fitted filter.
        """

        # Preprocess data
        X, y = np.asarray(X), np.asarray(y).flatten()
        n_features = X.shape[1]

        # Let's setup the result arrays
        # - Relevance with the target
        # - Partial score
        relevance = self.get_relevances(
            X,
            y,
            progress_bar=progress_bar,
            progress_bar_kwargs=dict(desc=f'CMIM: Relevance ({self.get_relevance_name()})'),
            n_jobs=n_jobs,
        )
        partial_score = relevance.copy()
        partial_score_comparisons = defaultdict(list)

        # Initialize selected features
        ranking = np.full(n_features, np.nan)

        n_iterations = self.get_n_iterations(n_features)
        for iteration in range(n_iterations):

            selected_features = np.where(~np.isnan(ranking))[0]
            remaining_features = np.where(np.isnan(ranking))[0]

            best_partial_score = 0

            # Compute the conditional of feature i with the selected features
            for i in tqdm(
                    remaining_features,
                    disable=not progress_bar,
                    desc=
                    f'CMIM: Conditional Relevance ({self.get_conditional_relevance_name()}) {iteration + 1}/{n_iterations}'
            ):

                # We compare against only features we haven't compared against yet
                selected_features_to_compare_against = set(selected_features) - set(partial_score_comparisons[i])

                for j in selected_features_to_compare_against:
                    # If the partial score is less than the best partial score, we can skip
                    if partial_score[i] > best_partial_score:
                        cond_relevance = self.get_conditional_relevance(X, y, i, j)
                        partial_score[i] = min(partial_score[i], cond_relevance)
                        partial_score_comparisons[i].append(j)

                if partial_score[i] > best_partial_score:
                    best_partial_score = partial_score[i]

            # Select the best feature only among the remaining features
            best_feature = np.argmax(np.where(np.isnan(ranking), partial_score, -np.inf))

            # Update results
            ranking[best_feature] = iteration + 1  # 1-indexed

        self.ranking_ = ranking
        self.relevance_ = relevance
        self.partial_score_ = partial_score

        return self