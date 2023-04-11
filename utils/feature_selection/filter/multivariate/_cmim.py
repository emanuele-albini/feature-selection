""" This module implements the CMIM (Conditional Mutual Information Maximization) 
    algorithm for feature selection.
"""
__all__ = ['CMIM']

import time
from typing import Union, Optional
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator
from ._utils import RankingSelectorMixin, RelevanceMixin, ConditionalRelevanceMixin
from ....parallel import from_numpy_to_shared_array, from_shared_array_to_numpy


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
        n_jobs: int = 1,
        progress_bar: bool = False,
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
        # conditional_relevances[i, j] = MI(Y, X_i  X_j)
        conditional_relevances = np.full((n_features, n_features), np.nan)

        # Initialize selected features
        ranking = np.full(n_features, np.nan)

        n_iterations = self.get_n_iterations(n_features)

        conditional_relevances_shared, conditional_relevances = from_numpy_to_shared_array(conditional_relevances,
                                                                                           return_numpy=True)
        X_splitted_shared = [from_numpy_to_shared_array(X[:, i].copy(), raw=True) for i in range(n_features)]
        y_shared = from_numpy_to_shared_array(y, raw=True)

        for iteration in range(n_iterations):

            selected_features = np.where(~np.isnan(ranking))[0]
            remaining_features = np.where(np.isnan(ranking))[0]
            remaining_features_shared, remaining_features = from_numpy_to_shared_array(remaining_features,
                                                                                       return_numpy=True,
                                                                                       raw=True)

            with Pool(
                    processes=n_jobs,
                    initializer=self._init_pool_process,
                    initargs=(
                        conditional_relevances_shared,
                        conditional_relevances.shape,
                        conditional_relevances.dtype,
                        X_splitted_shared,
                        (X.shape[0], ),
                        X.dtype,
                        y_shared,
                        y.shape,
                        y.dtype,
                        remaining_features_shared,
                        remaining_features.shape,
                        remaining_features.dtype,
                    ),
            ) as pool:  # Pool will be closed automatically

                # Generate the list of conditional relevances to compute
                ijs = []
                for i in remaining_features:
                    # We compute conditional relevances only with the selected features
                    # And only for features for which we don't have the conditional relevance yet
                    selected_features_to_compare_against = [
                        j for j in selected_features if np.isnan(conditional_relevances[i, j])
                    ]
                    ijs.extend([(i, j) for j in selected_features_to_compare_against])

                # Compute the conditional relevances in parallel
                # imap returns an iterator, so we need to wrap it in a list to force the computation
                _ = list(
                    tqdm(
                        pool.imap(
                            self._compute_conditional_relevance,
                            ijs,
                        ),
                        disable=not progress_bar,
                        total=len(ijs),
                        desc=
                        f'CMIM: Conditional Relevance ({self.get_conditional_relevance_name()}) {iteration + 1}/{n_iterations}'
                    ))

                # Select the best feature only among the remaining features
                partial_scores = conditional_relevances.min(axis=1)
                best_feature = np.argmax(np.where(np.isnan(ranking), partial_scores, -np.inf))

                # Update results
                ranking[best_feature] = iteration + 1  # 1-indexed

        self.ranking_ = ranking
        self.relevance_ = relevance
        self.partial_score_ = partial_scores

        return self

    def _compute_conditional_relevance(self, args):
        i, j = args

        # Compute the current partial scores (may change by the time we compute the conditional relevance)
        partial_scores = conditional_relevances.min(axis=1)

        # We skip the feature if the partial score is less than the best partial score
        # It cannot become the best feature
        if partial_scores[i] > partial_scores[remaining_features].max():
            return

        # Compute the conditional relevance
        cond_relevance = self._get_conditional_relevance(X[i], y, X[j], i, j)

        # Save it to the shared array
        conditional_relevances_shared.acquire()
        conditional_relevances[i, j] = cond_relevance
        conditional_relevances_shared.release()

    def _init_pool_process(
        self,
        conditional_relevances_shared_,
        conditional_relevances_shape,
        conditional_relevances_dtype,
        X_splitted_shared,
        X_shape,
        X_dtype,
        y_shared,
        y_shape,
        y_dtype,
        remaining_features_shared,
        remaining_features_shape,
        remaining_features_dtype,
    ):
        """Initialize a process in the pool."""
        global conditional_relevances, conditional_relevances_shared, X, y, remaining_features

        t = time.time()

        # Get the shared arrays in numpy format
        conditional_relevances = from_shared_array_to_numpy(
            conditional_relevances_shared_,
            shape=conditional_relevances_shape,
            dtype=conditional_relevances_dtype,
        )
        remaining_features = from_shared_array_to_numpy(
            remaining_features_shared,
            shape=remaining_features_shape,
            dtype=remaining_features_dtype,
        )
        X = [from_shared_array_to_numpy(x_shared, shape=X_shape, dtype=X_dtype) for x_shared in X_splitted_shared]
        y = from_shared_array_to_numpy(y_shared, shape=y_shape, dtype=y_dtype)
        print("Took", time.time() - t, "to convert to NumPy...")

        conditional_relevances_shared = conditional_relevances_shared_