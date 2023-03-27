import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utils.feature_selection.model_store import ModelStore
from utils.file import save_model, make_booster_agnostic
from utils.file import (
    save_yaml,
    load_yaml,
    save_json,
    load_json,
    save_data,
    load_data,
    load_booster,
    save_code,
)


def train_model(X, y, random_state):
    model = XGBClassifier()
    model.fit(X, y)
    return model


def load_training_data(random_state, test_size=0.8):

    # Load the data
    dataset = fetch_california_housing(as_frame=True)
    X = dataset.data
    y = dataset.target

    # Binarize
    y = (y > np.quantile(y, .85)).astype(int)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                        y.values,
                                                        random_state=random_state,
                                                        shuffle=True,
                                                        test_size=test_size)
    features = X.columns.values
    X_train = pd.DataFrame(X_train, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)
    y_train = pd.Series(y_train, name='target')
    y_test = pd.Series(y_test, name='target')
    return X_train, X_test, y_train, y_test, features


load_model = load_booster


def train_and_save_model(X, y, path=None, params=None):
    # Train SMM XGBoost model and save it
    model = XGBClassifier(**(params or dict(
        n_estimators=10,
        max_depth=3,
        random_state=0,
        objective="binary:logistic",
    )))
    model.fit(X, y)
    if path is not None:
        save_model(model, path)

    # Extract the booster
    booster = model.get_booster()
    booster = make_booster_agnostic(booster)

    return booster


def create_model_store(path):
    """Create a model store at the given path.

    Args:
        path (str): Path to the model store.

    Returns:
        ModelStore : Model store.
    """
    return ModelStore(
        training_routine=lambda X, y, key: train_and_save_model(X, y, f'{path}/{key}'),
        load_routine=lambda key: load_model(f'{path}/{key}'),
    )
