__author__ = "Emanuele Albini"
__all__ = [
    'load_data',
    'save_data',
    'save_dataframe',
    'load_dataframe',
]

import os
import time
import logging
import numpy as np
import pandas as pd
from ._utils import _add_extension_to_filename, _remove_extension_from_filename, _infer_format

DEFAULT_FORMAT = 'csv'
FORMATS = ['csv', 'parquet']


def save_dataframe(df, filename, format=None):
    if format is None:
        format = _infer_format(filename, formats=FORMATS, default=DEFAULT_FORMAT)

    filename = _add_extension_to_filename(filename, format)

    logging.info(f'Saving dataframe to {filename} ...')
    start = time.perf_counter()

    if format == 'csv':
        df.to_csv(filename)
    elif format == 'parquet':
        df.to_parquet(filename)
    else:
        raise ValueError(f'Format {format} not supported.')

    logging.info(f'Dataframe save in {np.round(time.perf_counter()-start, 4)} seconds.')


def load_dataframe(filename, format=None):
    # Infer format
    if format is None:
        format = _infer_format(filename, formats=FORMATS, default=DEFAULT_FORMAT)

    # Add extension if necessary
    if not os.path.exists(filename):
        filename = _add_extension_to_filename(filename, format)

    logging.info('Loading dataframe from %s...', filename)
    start = time.perf_counter()

    if format == 'csv':
        data = pd.read_csv(filename, header=0, index_col=0)
    elif format == 'parquet':
        data = pd.read_parquet(filename)
    else:
        raise ValueError(f'Format {format} not supported.')

    logging.info(f'Load done in {np.round(time.perf_counter()-start, 4)} seconds.')
    return data


def save_data(X, filename, format=None):
    if format is None:
        format = _infer_format(filename, formats=FORMATS, default=DEFAULT_FORMAT)

    # Remove extension from the filename if it was passed
    filename = _remove_extension_from_filename(filename, format)

    # We discriminate the datatype based on the extension
    if isinstance(X, pd.DataFrame):
        save_dataframe(X, filename + f'.{format}', format=format)
    elif isinstance(X, pd.Series):
        save_dataframe(pd.DataFrame(X), filename + f'.series.{format}', format=format)
    elif isinstance(X, np.ndarray):
        if len(X.shape) == 2:
            save_dataframe(pd.DataFrame(X), filename + f'.npy.{format}', format=format)
        elif len(X.shape) == 1:
            save_dataframe(pd.DataFrame(X), filename + f'.npy.series.{format}', format=format)
        else:
            raise NotImplementedError(f'Unsupported NumPy array shape: {X.shape}')
    else:
        raise NotImplementedError(f'Unsupported data type: {type(X)}')


def load_data(filename, format=None):
    # Infer format
    if format is None:
        format = _infer_format(filename, formats=FORMATS, default=DEFAULT_FORMAT)

    filename = _remove_extension_from_filename(filename, format)

    # We discriminate the datatype based on the extension
    if os.path.exists(filename + f'.npy.{format}'):  # Numpy 2D
        return load_dataframe(filename + f'.npy.{format}', format=format).values
    elif os.path.exists(filename + f'.npy.series.{format}'):  # Numpy 1D
        return load_dataframe(filename + f'.npy.series.{format}', format=format).values[:, 0]
    elif os.path.exists(filename + f'.series.{format}'):  # Series
        return load_dataframe(filename + f'.series.{format}', format=format).iloc[:, 0]
    else:  # DataFrame
        return load_dataframe(filename, format=format)
