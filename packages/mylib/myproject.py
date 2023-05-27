import json
import os, sys
import statistics

from typing import List
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import contextlib
import time
import math
import numpy as np
from . import myevaluator

#from packages import bicpy
from packages.bicpy import gene2pattern, bicpy
from packages.mymemoize import memoize
from packages.tcgahandler import LayerDataset
from packages.pydge import pydge
from sklearn.model_selection import StratifiedKFold


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, make_scorer
import sklearn 

from IPython.display import display
import umap

#from mapping2 import Mapper

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


RS = 42

classifiers_dict = {"logistic_regression": LogisticRegression(solver="liblinear"),
                    "elastic_net": LogisticRegression(solver="saga", penalty="elasticnet", max_iter=200, l1_ratio=0.7, tol=0.01),
                    "svm": SVC(),
                    "nb": GaussianNB(),
                    "random_forest": RandomForestClassifier(random_state=42)}

metrics_dict = {"accuracy": make_scorer(accuracy_score),
                "recall": make_scorer(recall_score, zero_division=0, average='macro'),
                "precision": make_scorer(precision_score, zero_division=0, average='macro'),
                "f1_score": make_scorer(f1_score, zero_division=0, average='macro')}

# default_param = {
#     'symmetries': False,
#     'normalization': "column",
#     'discretization': "normal_distribution",
#     'noise_relaxation': "optional",
#     'filling_criteria': "remove",
#     'pattern_type': "constant",
#     'orientation': "rows",
#     'remove_percentage': 0.1, # with only 1 iter doesn't affect
# }

discretize_parameters = ["nr_labels", "symmetries", "normalization", "discretization", "noise_relaxation", "filling_criteria"]


def confirm_create_dir(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)



#######################################################
def transform_and_evaluate_validation(parameterization: dict, data: pd.DataFrame, target: str,
                                      classifier_list: List[str], metric_list: List[str], normalization: bool, 
                                      n_folds: int, filtering: str, filter_by: str, 
                                      balancing: bool, distance="euclidean",verbose=0, return_shapes=False) -> pd.DataFrame:
    """"""
    variables =  data.drop(columns=[target]).columns.tolist()
    if verbose>0:
        print("pre-transformation pattern data shape: ", data.shape)

    y = data[target].values
    x = data[variables].values

    # define dict to save results
    result_dict = {"dataset": [], "shape": [], "classifier": []}
    for metric in metric_list:
        result_dict[metric] = []

    shapes = []
    skf = StratifiedKFold(n_splits=n_folds, random_state=RS, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        df = data.copy()
        train_data = df.iloc[train_index, :]

        if normalization:
            scaler = StandardScaler().fit(train_data[variables]) # fit only on trainning data
            scaled_data = scaler.transform(df[variables])      # transform whole dataset
            scaled_data = pd.DataFrame(scaled_data, columns=variables, index=df.index) #turn back to DataFrame
            scaled_data[target] = df[target] # add back target variable
            df = scaled_data


        discretize_parameterization = {key: value for key, value in parameterization.items() if key in discretize_parameters}
        discrete_data, intscores = bicpy.discretize_data(df, discretize_parameterization, verbose)         
        discrete_data[target] = df[target]
        train_discrete =  discrete_data.iloc[train_index, :]
        intscores_train = []
        for i in range(len(intscores)):
            if i in train_index:
                intscores_train.append(intscores[i])


        t = time.time()
        patterns = bicpy.run(parameterization,  train_discrete, discretize=False, intscores=intscores_train, verbose=verbose)
        # transform whole discretized dataset using the patterns found on the trainning data 
        data_pattern = gene2pattern.create_pattern_dataset(discrete_data, target, patterns, parameterization,
                                                           filtering=filtering, filter_by=filter_by, discretize=False,distance=distance, 
                                                           verbose=verbose)
        shapes.append(data_pattern.shape)

        # divide in train and test, X and y
        train_pattern = data_pattern.iloc[train_index, : ]
        test_pattern = data_pattern.iloc[test_index, : ]
        y_train = train_pattern[target].values
        x_train = train_pattern.drop(columns=[target]).values
        y_test = test_pattern[target].values
        x_test = test_pattern.drop(columns=[target]).values
        
        # train and test classifiers
        for clf_key in classifier_list:
            result_dict["dataset"].append("pattern")
            result_dict["shape"].append(data_pattern.shape)
            result_dict["classifier"].append(clf_key)
            clf = sklearn.base.clone(classifiers_dict[clf_key])   # copy of classifier that has not been fitted
            clf.fit(x_train, y_train)
            for metric in metric_list:
                    m_func = metrics_dict[metric]
                    res = m_func(clf, x_test, y_test)
                    # complete row with metrics
                    result_dict[metric].append(res)
 
    results = pd.DataFrame(result_dict)
    results = results.melt(id_vars=["dataset", "shape", "classifier"], value_name="Score", var_name="Metric")

    dims = [s[1] for s in shapes]
    print("Pattern dimensions: ", dims, "Mean: ", statistics.fmean(dims))
    if return_shapes:
        return results, shapes
    else:
        return results


def evaluate_validation(data: pd.DataFrame, counts: pd.DataFrame, target: str, pvalue: float, classifier_list: List[str],
                       metric_list: List[str], normalization: bool, n_folds: int, balancing: bool, n_genes = None, verbose=0):
    
    variables = data.drop(columns=[target]).columns.tolist()
    y = data[target].values
    x = data[variables].values

    result_dict = {"dataset": [], "shape": [], "classifier": []}
    for metric in metric_list:
        result_dict[metric] = []

    dimensions = []

    skf = StratifiedKFold(n_splits=n_folds, random_state=RS, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        #print(f"\n## FOLD {i+1} - evaluate_validation\t###")
        df = data.copy()
        # DEG analysis
        if counts is not None: # not protein layer
            train_counts = counts.iloc[train_index, :]
            variables = pydge.deg_filtering(train_counts, target, pvalue = pvalue, n_genes=n_genes)
            #print(f"DEG reduced {len(df.columns)-1} to {len(variables)}")
            dimensions.append(len(variables))
            df = df[variables + [target]]
        
        train_data = df.iloc[train_index, :]

        # normalization
        if normalization:
            scaler = StandardScaler().fit(train_data[variables]) # fit only on trainning data
            scaled_data = scaler.transform(df[variables])      # transform whole dataset
            scaled_data = pd.DataFrame(scaled_data, columns=variables, index=df.index) #turn back to DataFrame
            scaled_data[target] = df[target] # add back target variable
            df = scaled_data

        # divide in train and test, X and y
        train_data = df.iloc[train_index, : ]
        test_data = df.iloc[test_index, : ]
        y_train = train_data[target].values
        x_train = train_data.drop(columns=[target]).values
        y_test = test_data[target].values
        x_test = test_data.drop(columns=[target]).values
        
        # train and test classifiers
        for clf_key in classifier_list:
            result_dict["dataset"].append("normal")
            result_dict["shape"].append(df.shape)
            result_dict["classifier"].append(clf_key)
            clf = sklearn.base.clone(classifiers_dict[clf_key])   # copy of classifier that has not been fitted
            clf.fit(x_train, y_train)
            for metric in metric_list:
                    m_func = metrics_dict[metric]
                    res = m_func(clf, x_test, y_test)
                    # complete row with metrics
                    result_dict[metric].append(res)

    results = pd.DataFrame(result_dict)
    results = results.melt(id_vars=["dataset", "shape", "classifier"], value_name="Score", var_name="Metric")
    if len(dimensions) > 0:
        print("DGE dimensions: ", dimensions, "Mean: ", statistics.fmean(dimensions))
    return results


def reduce_and_evaluate_validation(data, target, n_dimensions, reducer_key, classifier_list: List[str],
                       metric_list: List[str], normalization: bool, n_folds: int, balancing: bool, verbose=0):
    reducers = {"pca": PCA, "umap": umap.UMAP, "tsne": TSNE}
    variables = data.drop(columns=[target]).columns.tolist()
    y = data[target].values
    x = data[variables].values

    result_dict = {"dataset": [], "classifier": []}
    for metric in metric_list:
        result_dict[metric] = []

    skf = StratifiedKFold(n_splits=n_folds, random_state=RS, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        #print(f"\n## FOLD {i+1} - reduce_and_evaluate_validation\t###")
        df = data.copy()
        train_data = df.iloc[train_index, :]

        # normalization
        if normalization:
            scaler = StandardScaler().fit(train_data[variables]) # fit only on trainning data
            scaled_data = scaler.transform(df[variables])      # transform whole dataset
            scaled_data = pd.DataFrame(scaled_data, columns=variables, index=df.index) #turn back to DataFrame
            scaled_data[target] = df[target] # add back target variable
            df = scaled_data

        # UMAP reduction
        n_components = min(min(train_data.shape[0], len(variables)), n_dimensions)
        reducer = reducers[reducer_key.lower()](n_components=n_components)
        reducer =  reducer.fit(train_data[variables])
        reduced_data = reducer.transform(df[variables])
        cols = [f"D{i+1}" for i in range(reduced_data.shape[1])]
        reduced_data = pd.DataFrame(reduced_data, columns=cols, index=df.index) #turn back to DataFrame
        reduced_data[target] = df[target] # add back target variable
        df = reduced_data

        # divide in train and test, X and y
        train_data = df.iloc[train_index, : ]
        test_data = df.iloc[test_index, : ]
        y_train = train_data[target].values
        x_train = train_data.drop(columns=[target]).values
        y_test = test_data[target].values
        x_test = test_data.drop(columns=[target]).values
        
        # train and test classifiers
        for clf_key in classifier_list:
            result_dict["dataset"].append(reducer_key)
            result_dict["classifier"].append(clf_key)
            clf = sklearn.base.clone(classifiers_dict[clf_key])   # copy of classifier that has not been fitted
            clf.fit(x_train, y_train)
            for metric in metric_list:
                    m_func = metrics_dict[metric]
                    res = m_func(clf, x_test, y_test)
                    # complete row with metrics
                    result_dict[metric].append(res)

    results = pd.DataFrame(result_dict)
    results = results.melt(id_vars=["dataset", "classifier"], value_name="Score", var_name="Metric")
    return results

######################################################################################################