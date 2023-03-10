__author__ = 'Emanuele Albini'
__all__ = [
    'save_booster',
    'load_booster',
    'load_booster_params',
    'make_booster_agnostic',
]

import os
import time
import random
import json
import logging
import numpy as np
from ._json import load_json, save_json
from ._utils import _remove_extension_from_filename

BOOSTER_SUFFIX = '.bin'
BOOSTER_TXT_SUFFIX = '.bin.txt'
BOOSTER_PARAMS_SUFFIX = '.bin.json'

FORMATS = ['bin.txt', 'bin.json', 'binary', 'bin', 'json', 'txt']

DEFAULT_XGBOOST_NTHREAD = 24


def save_booster(model, path):
    """Save the booster model in 3 files:
        - .bin > Booster file (agnostic to feature names)
        - .bin.json > Booster parameters
        - .bin.txt > Booster textual description (agnostic to feature names)

    Args:
        model (xgboost.Booster): The model
        path (str): The path without extension or '.bin'
    """
    path = _remove_extension_from_filename(path, formats=FORMATS)

    # Create directory if not exists
    dir_name = os.path.dirname(path)
    if dir_name != '':
        os.makedirs(dir_name, exist_ok=True)

    # Dump textual description of the trees (with feature names)
    model.dump_model(path + BOOSTER_TXT_SUFFIX)

    # Dump raw booster parameters
    save_json(json.loads(model.save_config()), path + BOOSTER_PARAMS_SUFFIX)

    # Make the model agnostic to feature names
    model = make_booster_agnostic(model)

    # Dump booster object
    logging.info(f'Saving model in {path + BOOSTER_SUFFIX} ...')
    model.save_model(path + BOOSTER_SUFFIX)


def load_booster_params(path):
    if path.endswith('.bin'):
        path = path[:-4]

    path = path + BOOSTER_PARAMS_SUFFIX
    if not os.path.exists(path):
        raise FileNotFoundError(f'Booster params not found: {path}')

    return load_json(path)


def load_booster(path):
    from xgboost import Booster

    if path.endswith('.bin'):
        path = path[:-4]

    path = path + BOOSTER_SUFFIX
    if not os.path.exists(path):
        raise FileNotFoundError(f'Booster file not found: {path}')

    booster = Booster(model_file=path)

    return booster


def make_booster_agnostic(booster):
    from xgboost import Booster

    temp_file = f'temp-{time.perf_counter()}-{random.random()}.bin'
    booster.save_model(temp_file)
    booster = Booster(model_file=temp_file)
    os.remove(temp_file)

    return booster


# def prepare_booster(booster, **kwargs):
#     booster.set_param({'nthread': DEFAULT_XGBOOST_NTHREAD})  # This has been deprecated (but we keep it for compatibility)
#     booster.set_param({'n_jobs': DEFAULT_XGBOOST_NTHREAD})
#     return booster
