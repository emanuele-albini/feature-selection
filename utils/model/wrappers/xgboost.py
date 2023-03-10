__all__ = [
    'XGBClassifierSKLWrapper',
]
__author__ = 'Emanuele Albini'

from typing import Optional, Union
from cached_property import cached_property

import numpy as np
import xgboost

from sklearn.base import BaseEstimator, MetaEstimatorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from ...file import (
    load_booster,
    save_booster as _save_booster,
    load_wrapper_params, 
    save_wrapper_params,
    load_wrapper_attributes, 
    save_wrapper_attributes, 
    make_booster_agnostic,
)
from ..tune import find_best_threshold_by_rate_equivalence


class XGBClassifierSKLWrapper(BaseEstimator, ClassifierMixin):
    """
        Model wrapper that provide the `predict` and `predict_proba` interface
        for compatibility with all methods expecting SKLEARN-like output
    """

    PARAMS_SUFFIX = '.wrapper.json'

    _required_parameters = ["booster"]

    def __init__(
        self,
        booster: xgboost.Booster,
        *,
        ntree_limit: int = 0,
        threshold: float = .5,
        missing: Optional[Union[float, int, bool]] = None,
    ):
        """Initialize Wrapper

        Args:
            booster (xgboost.Booster): Booster
            threshold (float, optional): The decision threshold (P > thr => C = 1). Defaults to .5.
            missing (Optional[Union[float, int, bool]], optional): Missing value symbol in the data for xgboost.DMatrix. Defaults to None (NaN).
        """

        self.booster = booster
        self.ntree_limit = ntree_limit
        self.missing = missing
        self.threshold = threshold

    def save(self, path, *args, **kwargs):
        return save_model(self, path, *args, **kwargs)

    def save(self, path, save_booster=True):
        # Save params
        params = self.get_params()
        if 'booster' in params:
            del params['booster']
        save_wrapper_params(params, path)

        # Save booster
        if save_booster:
            _save_booster(self.get_booster(), path)

        # Save fitted attributes
        save_wrapper_attributes(self, path)

    @classmethod
    def load(cls, path):
        """Load the model from a file

        Args:
            path (str): The path where the model is saved

        Returns:
            XGBClassifierSKLWrapper: The model (wrapper)
        """
        # Load params & booster
        params = load_wrapper_params(path)
        booster = load_booster(path)
        attributes = load_wrapper_attributes(path, safe=True)

        # Filter params (keep only those of the wrapper)
        params_names = set(list(cls(xgboost.Booster()).get_params().keys())) - {'booster'}
        params = {k: v for k, v in params.items() if k in params_names}

        model = cls(booster, **params)
        for k, v in attributes.items():
            setattr(model, k, v)

        return model

    def get_threshold(self):
        try:
            check_is_fitted(self)
            return self.threshold_
        except NotFittedError as e:
            if not isinstance(self.threshold, str):
                return self.threshold
            else:
                raise e

    def fit(self, X, y):
        if isinstance(self.threshold, str):
            if self.threshold == 'auto':
                self.threshold_ = find_best_threshold_by_rate_equivalence(self, X, y)
            else:
                raise ValueError(f'Invalid threshold {self.threshold}')
        else:
            self.threshold_ = self.threshold

        return self

    @cached_property
    def dmatrix_kwargs(self):
        kwargs = {}
        if hasattr(self, 'missing') and self.missing is not None:
            kwargs['missing'] = self.missing
        return kwargs

    def predict(self, X) -> np.ndarray:
        """
            Returns:
                [np.ndarray] : shape = (n_samples)
                The prediction (0 or 1), it returns 1 iff `probability of class 1 > threshold`
        """
        X = np.asarray(X)
        return (
            self.booster.predict(xgboost.DMatrix(X, **self.dmatrix_kwargs), ntree_limit=self.ntree_limit) >
            self.get_threshold()
        ) * 1

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X)
        return self.booster.predict(
            xgboost.DMatrix(X, **self.dmatrix_kwargs), ntree_limit=self.ntree_limit, output_margin=True
        )

    def predict_probs(self, X) -> np.ndarray:
        """
            Returns:
                [np.ndarray] : shape = (n_samples)
                It is the probability of class 1
        """
        X = np.asarray(X)
        return self.booster.predict(xgboost.DMatrix(X, **self.dmatrix_kwargs), ntree_limit=self.ntree_limit)

    def predict_proba(self, X, *args, **kwargs) -> np.ndarray:
        """
            Returns:
                [np.ndarray] : shape = (n_samples, 2)
                [:, 0] is the probability of class 0
                [:, 1] is the probability of class 1
        """
        ps = self.predict_probs(X, *args, **kwargs).reshape(-1, 1)
        return np.hstack([1 - ps, ps])

    def get_booster(self):
        """
        Returns:
            xgboost.Booster: Booster
        """
        return self.booster
