__author__ = "Emanuele Albini"
__all__ = [
    'save_model',
    'save_attributes',
    'save_wrapper_params',
    'save_wrapper_attributes',
    'load_params',
    'load_attributes',
    'load_wrapper_params',
    'load_wrapper_attributes',
]

import os
import time
import random
import json
import numpy as np
import logging
import functools
import warnings
from inspect import isclass

from ._json import load_json, save_json
from ._xgboost import save_booster, load_booster, FORMATS as XGBOOST_FORMATS
from ._utils import _remove_extension_from_filename, _add_extension_to_filename

FORMATS = [
    'wrapper.attributes.json', 'attributes.json', 'wrapper.json', 'wrapper', 'pkl', 'json', 'bin', 'pickle', 'binary'
]
FORMATS = list(set(FORMATS + XGBOOST_FORMATS))

PARAMS_SUFFIX = '.json'
ATTRIBUTES_SUFFIX = '.attributes.json'
WRAPPER_PARAMS_SUFFIX = '.wrapper.json'
WRAPPER_ATTRIBUTES_SUFFIX = '.wrapper.attributes.json'


def save_model(model, path, save_base_estimator=True, **kwargs):
    """Save the parameters (if passed) and the model (according to the type of model):
        - .json > Model parameters

    Args:
        model (A model): The model.
            It can be:
                - xgboost.Booster or 
                - XGBoost model wrapper that supports 'get_booster' and 'get_params'.
        path (str): The path without extension
    """
    # Create directory if not exists
    dir_name = os.path.dirname(path)
    if dir_name != '':
        os.makedirs(dir_name, exist_ok=True)

    path = _remove_extension_from_filename(path, formats=FORMATS)

    logging.info(f'Saving {type(model).__name__} to {path} ...')
    if type(model).__module__.startswith('xgboost') and type(model).__name__ == 'XGBClassifier':

        params = model.get_params()

        # Remove defaults
        params = {k: v for k, v in params.items() if v is not None}

        # Remove more defaults
        if 'missing' in params and np.isnan(params['missing']):
            del params['missing']

        if params is not None:
            params.update(params)
        params.update(kwargs)

        save_params(params, path)
        if save_base_estimator:
            save_booster(model.get_booster(), path)

    elif type(model).__module__.startswith('xgboost') and type(model).__name__ == 'Booster':
        save_booster(model, path)

    elif type(model).__name__ == 'XGBClassifierSKLWrapper':
        model.save(path, save_booster=save_base_estimator)
    else:
        raise NotImplementedError(f'Unsupported model type: {type(model).__name__}')

    logging.info(f'Done saving {type(model).__name__}.')


def _save_params(params, path, suffix):
    path = _remove_extension_from_filename(path, formats=FORMATS)
    path = _add_extension_to_filename(path, suffix)
    save_json(params, path)


def _load_params(path, suffix):
    path = _remove_extension_from_filename(path, formats=FORMATS)
    path = _add_extension_to_filename(path, suffix)
    return load_json(path)


def _save_attributes(model, path, suffix):
    path = _remove_extension_from_filename(path, formats=FORMATS)
    path = _add_extension_to_filename(path, suffix)

    attributes_names = __get_fitted_attribute_names(model)
    attributes = {name: getattr(model, name) for name in attributes_names}

    if len(attributes) > 0:
        save_json(attributes, path, serialize=True)


def _load_attributes(path, suffix, safe=False):
    path = _remove_extension_from_filename(path, formats=FORMATS)
    path = _add_extension_to_filename(path, suffix)

    if safe and not os.path.exists(path):
        warnings.warn(f'Could not load attributes from disk {path}. Returning an empty dict of attributes.')
        return {}
    else:
        return load_json(path)


def __get_fitted_attribute_names(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Returns the names of the fitted parameters of a scikit-learn model.
        This  function is based on sklearn.utils.validation.check_is_fitted(...)
    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        fitted = all_or_any([hasattr(estimator, attr) for attr in attributes])
    elif hasattr(estimator, "__sklearn_is_fitted__"):
        fitted = estimator.__sklearn_is_fitted__()
    else:
        fitted = [v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")]

    return fitted


load_params = functools.partial(_load_params, suffix=PARAMS_SUFFIX)
load_attributes = functools.partial(_load_attributes, suffix=ATTRIBUTES_SUFFIX)
load_wrapper_params = functools.partial(_load_params, suffix=WRAPPER_PARAMS_SUFFIX)
load_wrapper_attributes = functools.partial(_load_attributes, suffix=WRAPPER_ATTRIBUTES_SUFFIX)
save_params = functools.partial(_save_params, suffix=PARAMS_SUFFIX)
save_attributes = functools.partial(_save_attributes, suffix=ATTRIBUTES_SUFFIX)
save_wrapper_params = functools.partial(_save_params, suffix=WRAPPER_PARAMS_SUFFIX)
save_wrapper_attributes = functools.partial(_save_attributes, suffix=WRAPPER_ATTRIBUTES_SUFFIX)
