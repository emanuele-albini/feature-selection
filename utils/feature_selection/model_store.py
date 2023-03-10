__author__ = "Emanuele Albini"
__all__ = [
    'ModelStore',
]

import os
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class ModelStore(BaseEstimator):
    """A class that stores a model or trains it if it does not exist"""
    def __init__(self, training_routine, load_routine):
        """
        Args:
            training_routine (callable): A function that trains a model of the kind: f(X, y, key, *args, **kwargs) -> model
            load_routine (callable): A function that loads a model of the kind: f(key) -> model
        """

        self.training_routine = training_routine
        self.load_routine = load_routine

    def fit(self, X, y, *args, support: np.ndarray = None, **kwargs):
        """Fit (or load) a model on the given data and save it using only the features in the support

        Args:
            X, y: Train data
            support (np.ndarray, optional): Features to use to train the model. Defaults to None (all features).
            It can be:
            - A boolean array of shape (n_features, )
            - An integer array of shape (n_selected_features, )
            - A string array of shape (n_selected_features, ) containing the names of the selected features
        """

        if support is None:
            support = np.arange(X.shape[1])

        support = np.asarray(support)

        # Convert boolean array to integer array
        if isinstance(support, np.ndarray) and support.dtype == np.bool:
            support = np.arange(X.shape[1])[support]
        # Convert string array to integer array
        elif isinstance(support, np.ndarray) and support.dtype == np.str:
            support = np.arange(X.shape[1])[np.in1d(X.columns.values, support)]

        # Check that the support is valid
        if not isinstance(support, np.ndarray) or support.dtype != np.int:
            raise ValueError(f"Invalid support: {support} (not Numpy integers)")
        if np.any(support < 0) or np.any(support >= X.shape[1]):
            raise ValueError(f"Invalid support: {support} - Feature ID < 0, or >= {X.shape[1]} (nb_features)")
        if len(support) != len(np.unique(support)):
            raise ValueError(f"Invalid support: {support} (not unique)")

        # Sort the support
        support = np.sort(support)

        # Create the model key
        key = "-".join([str(i) for i in support])
        logging.info(f'Querying for model {key} ...')

        # Try to load the model
        try:
            model = self.load_routine(key)
            logging.info(f'STORE: Successfully loaded {key} from the store.')
        except FileNotFoundError:
            logging.info(f'STORE: Could not load the model {key}. Retraining ...')
            model = None

        # If the model does not exist, train it (and save it)
        if model is None:
            if isinstance(X, pd.DataFrame):
                X_ = X.iloc[:, support]
            else:
                X_ = X[:, support]

            model = self.training_routine(X_, y, key, *args, **kwargs)

        return model
