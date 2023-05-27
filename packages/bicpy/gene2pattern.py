from math import ceil
from operator import itemgetter
import time
import math

import pandas as pd
import numpy as np
from typing import List

from sklearn.metrics import pairwise_distances

from . import bicpy

RANDOM_STATE = 42

filter_by_dict = {
    'lift': lambda pattern_list: {key: max(pattern.lift) for (key, pattern) in enumerate(pattern_list)},
    'support': lambda pattern_list: {key: len(pattern.rows) for (key, pattern) in enumerate(pattern_list)}
}

filtering_dict = {
    'q5': lambda values: np.quantile(values, 0.95),
    'q10': lambda values: np.quantile(values, 0.90),
    'q1': lambda values: np.quantile(values, 0.75),
    'q2': lambda values: np.quantile(values, 0.50),
    'q3': lambda values: np.quantile(values, 0.25),
    'number': lambda values_dict, n: dict(sorted(values_dict.items(), key=itemgetter(1), reverse=True)[:n])
}

discretize_parameters = ["nr_labels", "symmetries", "normalization", "discretization", "noise_relaxation", "filling_criteria"]


def norm_euclidean(x, z) -> float:
    euclidean = pairwise_distances(x, z, metric='euclidean')
    norm_euclidean = np.sqrt( np.power(euclidean, 2) / x.shape[1] )
    return norm_euclidean

# CREATE PATTERN-BASED DATASET #########################################################################################
def create_pattern_dataset(data: pd.DataFrame, target: str, patterns: List[bicpy.Pattern], parameterization: dict,
                           distance="euclidean", filtering: str = None, filter_by: str = None, discretize=True,
                            verbose=1):
    """Creates pattern based dataset from dataset in {data_path} and the Patterns received

    {parameterization} argument is necessary to create a discrete dataset as an intermediary step with the same
    parameters as when the Patterns were calculated
    Since the  discretization process removes the target variable, it is saved before this process and later appended
    """
    # print('begin create pattern dataset')
    target_column = data[target]
    # print('bicpy discretize data')
    if discretize:
        discretize_parameterization = {key: value for key, value in parameterization.items() if key in discretize_parameters}
        discrete_df, _ = bicpy.discretize_data(data, discretize_parameterization, verbose)
    else:
        discrete_df = data.drop(columns=[target])

    column_names = []
    columns = []
    if filtering is not None:
        patterns = filter_patterns(patterns, filtering, filter_by)
    
    for i, pattern in enumerate(patterns):
        col = create_pattern_column(discrete_df, pattern.columns, pattern.values, distance)
        columns.append(col)
        column_names.append(f"p{i + 1}")
    values = np.concatenate(columns, axis=1)
    pattern_based_df = pd.DataFrame(values, index=discrete_df.index, columns=column_names)
    pattern_based_df[target] = target_column
    #print("end create pattern dataset")
    return pattern_based_df


def create_pattern_column(df: pd.DataFrame, pattern_columns: List[str], pattern_values: list,
                          metric: str) -> np.ndarray:
    """Creates new column with distance {metric} between the values in columns {pattern_columns} and {pattern_values}

    {metric} can be one of the following: 'euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis',
    'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
    'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
    'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine', or 'precomputed'
    """

    old_values = df.loc[:, pattern_columns].values
    pattern_values = np.array(pattern_values).reshape(1, -1)
    if metric == "norm_euclidean":
        new_values = norm_euclidean(old_values, pattern_values)
    else:
        new_values = pairwise_distances(old_values, pattern_values, metric=metric)
    return new_values


def filter_patterns(patterns: List[bicpy.Pattern], type_of_filtering: str, filter_by: str, nr_patterns: int = None) -> List[bicpy.Pattern]:
    """When using quantiles, actually does it in a set, only unique values"""

    try:
        get_values_to_filter = filter_by_dict[filter_by]
    except KeyError:
        raise ValueError(f'Parameter {{filter_by}} "{filter_by}" is not valid. '
                         f'Must be one of the following: {list(filter_by_dict.keys())}.')

    try:
        filtering_func = filtering_dict[type_of_filtering]
    except KeyError:
        raise ValueError(f'Parameter {{type_of_filtering}} "{type_of_filtering}" is not valid. '
                         f'Must be one of the following: {list(filtering_dict.keys())}.')

    # replace lift None for 0
    patterns = replace_lift(patterns, 0)
    values_dict = get_values_to_filter(patterns)
    
    if type_of_filtering == 'number':
        if not isinstance(nr_patterns, int) or nr_patterns<1: raise ValueError(f'nr_patterns should be positive int but is {nr_patterns}, {type(nr_patterns)}')
        filtered_dict = filtering_func(values_dict, nr_patterns)
    else:
        values = list(values_dict.values())
        filtering_cutoff = filtering_func(values)
        filtered_dict = {k: v for (k, v) in values_dict.items() if v >= filtering_cutoff}

    filtered_indices = list(filtered_dict.keys())
    filtered_patterns = list(itemgetter(*filtered_indices)(patterns))  # get items from list using multiple indices
    return filtered_patterns


def replace_lift(patterns: List[bicpy.Pattern], replace: float) -> List[bicpy.Pattern]:
    for i in range(len(patterns)):
        if patterns[i].lift is None:
            patterns[i].lift = [replace, replace]
    return patterns
