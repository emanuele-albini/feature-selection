import logging
import time
from os import makedirs
from collections import defaultdict
from multiprocessing import cpu_count
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from utils.args import parse_args
from utils.feature_selection.filter.multivariate import FCBF, MRMR, CMIM
from my import (
    # get_features,
    load_training_data,
    load_data,
    save_data,
    save_model,
    save_yaml,
    save_code,
    create_model_store,
)


class SelectKBest:
    """
        This is a minimal re-implementaion of sklearn.feature_selection.SelectKBest with the following changes:
        - It supports passing NaN (even if the scoring function suppports NaN)
        - k is passed to get_support and not to the constructor
    """
    def __init__(self, scoring_function):
        self.scoring_function = scoring_function

    def fit(self, X, y):
        scores = self.scoring_function(X, y)
        if isinstance(scores, tuple):
            scores, pvalues = scores
        else:
            pvalues = np.full(len(scores), np.nan)

        self.scores_ = np.array(scores)
        self.pvalues_ = np.array(pvalues)
        return self

    def get_support(self, k):
        mask = np.zeros(len(self.scores_), dtype=bool)
        mask[np.argsort(self.scores_)[-k:]] = True
        return mask


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser(description="Feature Selection using Filter Methods")
    parser.add_argument('--name', type=str, help='Name of the model')
    parser.add_argument('--dir', type=str, default='results', help='Directory where to save the results')
    parser.add_argument('--region', type=str, default='EMEA', help='Region (EMEA, NA, ...)')
    parser.add_argument('--experiment', type=str, default=None, help='Experiment name')
    parser.add_argument('--date', type=str, default=None, help='Data date')
    parser.add_argument('--random_state', type=int, default=0, help='Random seed')
    parser.add_argument('--load', action='store_true', default=False, help='Load filters from disk')
    parser.add_argument('--train_size', type=float, default=1.0, help='Train Split')
    parser.add_argument('--k', nargs='+', type=int, help='Number of features to select (it can be a list)')
    parser.add_argument('--n_jobs', type=int, help='Number of jobs to run in parallel', default=cpu_count() - 1)
    args = parse_args(parser)

    assert args.name is not None
    assert isinstance(args.k, list)

    results_directory = f"{args.dir}/{args.experiment}/{args.region}/{args.name}"
    makedirs(results_directory, exist_ok=True)

    # Save configuration
    save_code(["utils"], f"{results_directory}/code")
    save_yaml(args, f"{results_directory}/args")

    # Load the data
    # features = get_features(args.region)
    X_train, X_test, y_train, y_test, features = load_training_data(random_state=args.random_state,
                                                                    test_size=1 - args.train_size)

    # Repeat data
    X_train = pd.concat([X_train] * 100, axis=0)
    y_train = pd.concat([y_train] * 100, axis=0)

    print('Shapes', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    MI_ARGS = dict(
        n=100000,
        random_state=args.random_state,
        discrete_features=False,
        discrete_target=True,
    )

    FILTERS = {
        'mRMR_mi':
        MRMR(
            relevance='mi',
            redundancy='mi',
            relevance_kwargs=MI_ARGS,
            redundancy_kwargs=MI_ARGS,
            symmetric_redundancy=True,
        ),
        'FCBF_mi':
        FCBF(
            relevance='mi',
            redundancy='mi',
            relevance_kwargs=MI_ARGS,
            redundancy_kwargs=MI_ARGS,
            symmetric_redundancy=True,
        ),
        'CMIM':
        CMIM(
            relevance='mi',
            conditional_relevance='cmi',
            relevance_kwargs=MI_ARGS,
            conditional_relevance_kwargs=MI_ARGS,
        ),
    }

    if not args.load:
        for filter_name, filter in FILTERS.items():
            t = time.perf_counter()
            logging.info(f'Fitting {filter_name} multivariate filter.')
            filter.fit(X_train, y_train, progress_bar=True, n_jobs=args.n_jobs)
            logging.info(f'Fitting {filter_name} multivariate filter took {time.perf_counter() - t:.2f} seconds.')

        # Let's generate 3 dataframes where rows are filters and columns are features:
        # - rankings: rankings of the features (1-indexed, the lower the more important)
        # - supports: binary flags of active/non-active features
        rankings = pd.DataFrame({filter_name: filter.ranking_
                                 for filter_name, filter in FILTERS.items()},
                                index=features).T
        supports = pd.DataFrame(
            {(filter_name, k): filter.get_support(k)
             for filter_name, filter in FILTERS.items() for k in args.k},
            index=features).T

        # Save results
        save_data(rankings, f"{results_directory}/rankings")
        save_data(supports, f"{results_directory}/supports")

    else:
        # Load filters
        scores = load_data(f"{results_directory}/rankings")
        supports = load_data(f"{results_directory}/supports")

    store = create_model_store(f"{args.dir}/{args.experiment}/{args.region}/store")

    for (filter_name, k), support in supports.iterrows():
        print(f'Training model using {filter_name} filter and {k} features.')
        results_directory_ = f"{results_directory}/{filter_name}/{k}"
        makedirs(results_directory_, exist_ok=True)

        X_train_ = X_train.loc[:, support]

        # Save features used
        save_data(features[support], f"{results_directory_}/features")
        save_data(X_train_, f"{results_directory_}/X_train", format='parquet')
        save_data(y_train, f"{results_directory_}/y_train", format='parquet')

        # Train and save
        wrapper = store.fit(X_train.values, y_train.values.flatten(), support=support)
        save_model(wrapper, f"{results_directory_}/model")

    print('Done.')
