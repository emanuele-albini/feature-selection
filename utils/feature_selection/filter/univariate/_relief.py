__author__ = "Emanuele Albini"
__all__ = ['reliefF']

import numpy as np

from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit

NAN_NUM = -999999


class DummyScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


def _is_nan(x):
    return (x == NAN_NUM) | np.isnan(x)


def _reliefF_distance(x, y, continuous_features_mask, discrete_features_mask, x_missing_proba, y_missing_proba):
    """
    Compute the distance between two samples x and y.
    """
    # Missing handling
    missing_mask_x = _is_nan(x)
    missing_mask_y = _is_nan(y)
    missing_mask_any = missing_mask_x | missing_mask_y
    missing_mask_both = missing_mask_x & missing_mask_y
    missing_mask_x = missing_mask_x & ~missing_mask_both
    missing_mask_y = missing_mask_y & ~missing_mask_both

    discrete_features_mask = discrete_features_mask & ~missing_mask_any
    continous_features_mask = continuous_features_mask & ~missing_mask_any

    dist = np.zeros(x.shape[0])

    # Discrete distance (Hamming)
    dist += 1.0 * discrete_features_mask * (x != y)

    # Continuous distance (Manhattan)
    dist += continous_features_mask * np.abs(x - y)

    # Missing distance (Difference in conditional missing probabilities)
    dist += missing_mask_both.sum() * np.abs(x_missing_proba - y_missing_proba)
    dist += missing_mask_x * np.abs(x_missing_proba - 1 + y_missing_proba)
    dist += missing_mask_y * np.abs(y_missing_proba - 1 + x_missing_proba)

    return dist


def reliefF_distance(*args, **kwargs):
    return _reliefF_distance(*args, **kwargs).sum()


def reliefF(X, y, n=10000, k=10, random_state=0, discrete_feature_indexes=None, scaler=None, n_jobs=1, verbose=True):

    # Convert to numpy arrays
    X, y = np.asarray(X), np.asarray(y).flatten()

    # We must impute NaNs because sklearn has a bug (it raise exception even if the distance metric supports NaNs)
    assert (X == NAN_NUM).sum() == 0
    X = np.nan_to_num(X, nan=NAN_NUM)

    # No discrete features or scaler by default
    discrete_feature_indexes = discrete_feature_indexes or []
    scaler = scaler or DummyScaler()

    # Sample the dataset
    sample_indices = next(StratifiedShuffleSplit(train_size=n, random_state=random_state).split(X=X, y=y))[0]
    X = X[sample_indices]
    y = y[sample_indices]
    X = scaler.transform(X)

    # Divide the dataset by class
    classes = np.unique(y)
    X_ = {c: X[y == c] for c in classes}

    # Compute the probabilities of each class
    class_probabilities = {c: len(X_[c]) / len(X) for c in classes}

    # Genenerate the masks for discrete and continuous features
    discrete_features_mask = np.zeros(X.shape[1], dtype=bool)
    discrete_features_mask[discrete_feature_indexes] = True
    continuous_features_mask = ~discrete_features_mask

    # Generate the missing probabilities for each class and feature
    missing_probabilities = {c: _is_nan(X[y == c]).mean(axis=0) for c in classes}

    # Create the nearest neighbors structure
    nn = {(c, c_): NearestNeighbors(
        n_neighbors=k,
        metric=lambda x, y: reliefF_distance(
            x,
            y,
            continuous_features_mask=continuous_features_mask,
            discrete_features_mask=discrete_features_mask,
            x_missing_proba=missing_probabilities[c],
            y_missing_proba=missing_probabilities[c_],
        ),
        n_jobs=n_jobs,
    ).fit(X)
          for c in classes for c_ in tqdm(
              [c__ for c__ in classes if c__ >= c], desc=f'RELIEF-F NN (c={c})', disable=not verbose, unit='classes')}

    def _get_nn(c, c_):
        if c_ >= c:
            return nn[(c, c_)]
        else:
            return nn[(c_, c)]

    # Compute the weights
    weights = np.zeros(X.shape[1])

    # For all the points (segmented by class)
    for c in classes:

        # Add the feature-wise distance from the same class neighbors
        print(f'RELIEF-F Same class (c={c}) ...', end=' ', flush=True)
        same_neighbors = _get_nn(c, c).kneighbors(X_[c], return_distance=False)
        for i, same_neighbors_ in enumerate(same_neighbors):
            for j, same_neigh in enumerate(same_neighbors_):
                weights += _reliefF_distance(
                    X_[c][i],
                    same_neigh,
                    continuous_features_mask=continuous_features_mask,
                    discrete_features_mask=discrete_features_mask,
                    x_missing_proba=missing_probabilities[c],
                    y_missing_proba=missing_probabilities[c],
                )
        print('Done')

        # Subtract the feature-wise distance from the other class neighbors
        # (weighted by the probability of the other class)
        for c_ in tqdm([c__ for c__ in classes if c__ != c],
                       desc=f'RELIEF-F Other classes (c={c})',
                       disable=not verbose,
                       unit=' classes'):
            weight_ = class_probabilities[c_] / (sum(list(class_probabilities.values())) - class_probabilities[c])
            other_neighbors = _get_nn(c, c_).kneighbors(X_[c], return_distance=False)
            for i, other_neighbors_ in enumerate(other_neighbors):
                for j, other_neigh in enumerate(other_neighbors_):
                    weights -= weight_ * _reliefF_distance(
                        X_[c][i],
                        other_neigh,
                        continuous_features_mask=continuous_features_mask,
                        discrete_features_mask=discrete_features_mask,
                        x_missing_proba=missing_probabilities[c],
                        y_missing_proba=missing_probabilities[c_],
                    )

    # Higher weights = feature more different within the same class
    # Higher weights = feature more similar compared to the other classes
    # We have to take the negative.
    return -weights
