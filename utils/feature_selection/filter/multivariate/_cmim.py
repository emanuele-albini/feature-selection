""" This module implements the CMIM (Conditional Mutual Information Maximization) 
    algorithm for feature selection.
"""
__all__ = ['CMIM']

import time
from typing import Union, Optional
from collections import defaultdict
from multiprocessing import Pool, Value

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

        partial_scores = relevance.copy()

        # conditional_relevances[i, j] = MI(Y, X_i  X_j)
        conditional_relevances = np.full((n_features, n_features), np.nan)

        # Initialize selected features
        ranking = np.full(n_features, np.nan)

        n_iterations = self.get_n_iterations(n_features)

        X_splitted = [X[:, i].copy() for i in range(n_features)]

        # Create shared arrays
        conditional_relevances_shared, conditional_relevances = from_numpy_to_shared_array(
            conditional_relevances,
            return_numpy=True,
        )
        partial_scores_shared, partial_scores = from_numpy_to_shared_array(
            partial_scores,
            return_numpy=True,
        )
        X_splitted_shared = [from_numpy_to_shared_array(x, raw=True) for x in X_splitted]
        y_shared = from_numpy_to_shared_array(y, raw=True)

        for iteration in range(n_iterations):

            best_value = -np.inf
            selected_features = np.where(~np.isnan(ranking))[0]
            remaining_features = np.where(np.isnan(ranking))[0]

            # Generate the list of conditional relevances to compute
            ijs = []
            expected_nb_comparisons = np.full(n_features, np.nan)
            for i in remaining_features:
                # We compute conditional relevances only with the selected features
                # And only for features for which we don't have the conditional relevance yet
                selected_features_to_compare_against = [
                    j for j in selected_features if np.isnan(conditional_relevances[i, j])
                ]
                expected_nb_comparisons[i] = len(selected_features_to_compare_against)
                ijs.extend([(i, j) for j in selected_features_to_compare_against])

            nb_comparisons = np.zeros(n_features)

            # Create shared memory objects
            best_value = Value('d', best_value)
            remaining_features_shared, remaining_features = from_numpy_to_shared_array(
                remaining_features,
                return_numpy=True,
                raw=True,
            )
            expected_nb_comparisons_shared, expected_nb_comparisons = from_numpy_to_shared_array(
                expected_nb_comparisons,
                return_numpy=True,
                raw=True,
            )
            nb_comparisons_shared, nb_comparisons = from_numpy_to_shared_array(
                nb_comparisons,
                return_numpy=True,
            )
            pool = Pool(
                processes=n_jobs,
                maxtasksperchild=1,
                initializer=self._init_pool_process,
                initargs=(
                    conditional_relevances_shared,
                    conditional_relevances.shape,
                    conditional_relevances.dtype,
                    partial_scores_shared,
                    partial_scores.shape,
                    partial_scores.dtype,
                    nb_comparisons_shared,
                    nb_comparisons.shape,
                    nb_comparisons.dtype,
                    best_value,
                    X_splitted_shared,
                    (X.shape[0], ),
                    X.dtype,
                    y_shared,
                    y.shape,
                    y.dtype,
                    remaining_features_shared,
                    remaining_features.shape,
                    remaining_features.dtype,
                    expected_nb_comparisons_shared,
                    expected_nb_comparisons.shape,
                    expected_nb_comparisons.dtype,
                ),
            )

            # Compute the conditional relevances in parallel
            # imap returns an iterator, so we need to wrap it in a list to force the computation
            _ = list(
                tqdm(
                    pool.imap(
                        self._compute_conditional_relevance_parallel,
                        ijs,
                    ),
                    disable=not progress_bar,
                    total=len(ijs),
                    desc=
                    f'CMIM: Conditional Relevance ({self.get_conditional_relevance_name()}) {iteration + 1}/{n_iterations}'
                ))

            # Select the best feature only among the remaining features
            best_feature = np.argmax(np.where(np.isnan(ranking), partial_scores, -np.inf))

            # Update results
            ranking[best_feature] = iteration + 1  # 1-indexed

            pool.close()
            pool.join()

        self.ranking_ = ranking
        self.relevance_ = relevance
        self.conditional_relevances_ = conditional_relevances
        self.partial_scores_ = partial_scores

        return self

    def _compute_conditional_relevance_parallel(self, args):
        i, j = args

        partial_scores_shared.acquire()
        best_value.acquire()
        nb_comparisons_shared.acquire()

        nb_comparisons[i] += 1

        # We skip the feature if the partial score is less than the best partial score
        # It cannot become the best feature
        skip = (partial_scores[i] <= best_value.value)

        nb_comparisons_shared.release()
        best_value.release()
        partial_scores_shared.release()

        if skip:
            return

        xi = from_shared_array_to_numpy(X_splitted_shared[i], shape=X_shape, dtype=X_dtype)
        xj = from_shared_array_to_numpy(X_splitted_shared[j], shape=X_shape, dtype=X_dtype)

        # Compute the conditional relevance
        cond_relevance = self._get_conditional_relevance(xi, y, xj, i, j)

        # Save it to the shared array
        partial_scores_shared.acquire()
        best_value.acquire()
        nb_comparisons_shared.acquire()
        conditional_relevances_shared.acquire()

        conditional_relevances[i, j] = cond_relevance
        partial_scores[i] = min(partial_scores[i], conditional_relevances[i, j])

        # Update the best value
        if nb_comparisons[i] == expected_nb_comparisons[i]:
            best_value.value = max(best_value.value, partial_scores[i])

        conditional_relevances_shared.release()
        nb_comparisons_shared.release()
        best_value.release()
        partial_scores_shared.release()

    def _init_pool_process(
        self,
        conditional_relevances_shared_,
        conditional_relevances_shape,
        conditional_relevances_dtype,
        partial_scores_shared_,
        partial_scores_shape,
        partial_scores_dtype,
        nb_comparisons_shared_,
        nb_comparisons_shape,
        nb_comparisons_dtype,
        best_value_,
        X_splitted_shared_,
        X_shape_,
        X_dtype_,
        y_shared,
        y_shape,
        y_dtype,
        remaining_features_shared,
        remaining_features_shape,
        remaining_features_dtype,
        expected_nb_comparisons_shared,
        expected_nb_comparisons_shape,
        expected_nb_comparisons_dtype,
    ):
        """Initialize a process in the pool."""
        global conditional_relevances, partial_scores, nb_comparisons, y, remaining_features, expected_nb_comparisons
        global X_splitted_shared, X_shape, X_dtype
        global conditional_relevances_shared, partial_scores_shared, nb_comparisons_shared, best_value

        # Get the shared arrays in numpy format
        conditional_relevances = from_shared_array_to_numpy(
            conditional_relevances_shared_,
            shape=conditional_relevances_shape,
            dtype=conditional_relevances_dtype,
        )
        partial_scores = from_shared_array_to_numpy(
            partial_scores_shared_,
            shape=partial_scores_shape,
            dtype=partial_scores_dtype,
        )
        nb_comparisons = from_shared_array_to_numpy(
            nb_comparisons_shared_,
            shape=nb_comparisons_shape,
            dtype=nb_comparisons_dtype,
        )
        remaining_features = from_shared_array_to_numpy(
            remaining_features_shared,
            shape=remaining_features_shape,
            dtype=remaining_features_dtype,
        )
        expected_nb_comparisons = from_shared_array_to_numpy(
            expected_nb_comparisons_shared,
            shape=expected_nb_comparisons_shape,
            dtype=expected_nb_comparisons_dtype,
        )
        y = from_shared_array_to_numpy(y_shared, shape=y_shape, dtype=y_dtype)

        # We need some shared variables in the child processes
        X_splitted_shared, X_shape, X_dtype, = X_splitted_shared_, X_shape_, X_dtype_,
        conditional_relevances_shared = conditional_relevances_shared_
        partial_scores_shared = partial_scores_shared_
        nb_comparisons_shared = nb_comparisons_shared_
        best_value = best_value_
