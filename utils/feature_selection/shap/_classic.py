__author__ = "Emanuele Albini"
__all__ = [
    "SHAPMethod",
    "SHAPModelWrapper",
]

import logging
from enum import Enum
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator

from shap.maskers import Independent
from shap.explainers._explainer import Explainer

from ...preprocessing import sample_data


class SHAPMethod(Enum):
    SHAP = "shap"
    LOSS_SHAP = "loss"


SHAP_KWARGS = defaultdict(
    dict, {
        SHAPMethod.SHAP: dict(
            feature_perturbation="interventional",
            link="identity",
            model_output="raw",
        ),
        SHAPMethod.SHAP: dict(
            feature_perturbation="interventional",
            link="identity",
            model_output="log_loss",
        ),
    }
)


class SHAPModelWrapper(BaseEstimator):
    """Model Wrapper for SHAP-based feature selection
        This class provides few SHAP based methods:
        - Mean Absolute SHAP (Default)
        - LossSHAP (by passing `model_output = 'loss'`)
    """
    def __init__(
        self,
        estimator,
        method=SHAPMethod.SHAP,
        nb_samples_background=10000,
        nb_samples_foreground=10000,
        random_state=0,
    ):

        self.nb_samples_background = nb_samples_background
        self.nb_samples_foreground = nb_samples_foreground
        self.random_state = random_state
        self.method = method

        self._estimator = estimator
        self._feature_importances = None

    def fit(self, X, y):
        self._estimator.fit(X, y)

        # Sample background data
        X_background_ = Independent(X, max_samples=self.nb_samples_background)

        # Sample foreground data
        X_foreground_ = sample_data(
            X,
            n=self.nb_samples_foreground,
            random_state=self.random_state + 1,
            replace=False,
            safe=True,
        )

        # Setup explainer
        self._explainer = Explainer(self._estimator, X_background_, **SHAP_KWARGS[self.method])
        logging.info(f"Using {self._explainer.__class__.__name__} SHAP explainer.")

        # Compute feature importances
        if self.method == SHAPMethod.SHAP:
            self._feature_importances = np.abs(self._explainer.shap_values(X_foreground_)).mean()
        elif self.method == SHAPMethod.LOSS_SHAP:
            self._feature_importances = self._explainer.shap_values(X_foreground_).mean()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self
