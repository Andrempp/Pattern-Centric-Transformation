import math
import imblearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from typing import List

from sklearn.utils._testing import ignore_warnings

from packages.mymemoize import memoize

RANDOM_STATE = 42


class DatasetEvaluator:
    classifiers_dict = {"logistic_regression": LogisticRegression(),
                        "naive_bayes": GaussianNB(),
                        "knn": KNeighborsClassifier(),
                        "svm": SVC(),
                        "decision_tree": DecisionTreeClassifier(),
                        "random_forest": RandomForestClassifier(random_state=42)}

    # metrics_dict = {"accuracy": lambda y, prediction: accuracy_score(y, prediction),
    #                 "recall": lambda y, prediction: recall_score(y, prediction, zero_division=0),
    #                 "precision": lambda y, prediction: precision_score(y, prediction, zero_division=0),
    #                 "f1_score": lambda y, prediction: f1_score(y, prediction, zero_division=0)}

    # using make_scorer makes it easier for cross-validation and only slightly slower for holdout
    metrics_dict = {"accuracy": make_scorer(accuracy_score),
                    "recall": make_scorer(recall_score, zero_division=0, average='macro'),
                    "precision": make_scorer(precision_score, zero_division=0, average='macro'),
                    "f1_score": make_scorer(f1_score, zero_division=0, average='macro')}

    plot_types = {"barplot": sns.barplot, "lineplot": sns.lineplot}

    def __init__(self, df_list: List[pd.DataFrame] | dict, target_list: List[str], classifier_list: List[str],
                 metric_list: List[str]):

        if len(df_list) == 0:
            raise ValueError('Argument {df_list} is empty.')
        if isinstance(df_list, dict):
            df_tags, df_list = zip(*df_list.items())
        if isinstance(df_list, list):
            df_tags = list(range(len(df_list)))

        if len(df_list) != len(target_list):
            raise ValueError(f"Arguments {{df_list}} and {{target_list}} should of same length. Current lengths are "
                             f"{len(df_list)} and {len(target_list)}, respectively")
        self.df_list = df_list
        self.df_tags = df_tags
        self.target_list = target_list

        for classifier in classifier_list:
            if classifier not in DatasetEvaluator.classifiers_dict.keys():
                raise ValueError(f"Classifier '{classifier}' not supported. Must be one of the following: "
                                 f"{DatasetEvaluator.classifiers_dict.keys()}")
        self.classifier_list = classifier_list

        for metric in metric_list:
            if metric not in DatasetEvaluator.metrics_dict.keys():
                raise ValueError(f"Metric '{metric}' not suppoerted. Must be one of the following: "
                                 f"{DatasetEvaluator.metrics_dict.keys()}")
        self.metric_list = metric_list

    @staticmethod
    def available_classifiers():
        return list(DatasetEvaluator.classifiers_dict.keys())

    @staticmethod
    def available_metrics():
        return list(DatasetEvaluator.metrics_dict.keys())

    @ignore_warnings(category=ConvergenceWarning)
    def evaluate(self, normalize=True, evaluation_type='holdout', test_size=0.3, cv=None, balancing=False, melt=True, train_test_indexes=None) -> pd.DataFrame:

        #print('Running myevaluator.DatasetEvaluator.evaluate')
        # create dict to save results and later turn into DataFrame
        result_dict = {"dataset": [], "classifier": []}
        for metric in self.metric_list:
            result_dict[metric] = []

        # cycle for input datasets
        for i, (df, target) in enumerate(zip(self.df_list, self.target_list)):
            # transform bool or list of bool to bool
            norm = (isinstance(normalize, list) and normalize[i]) or (normalize is True)
            for classifier in self.classifier_list:
                result_dict["dataset"].append(self.df_tags[i])
                result_dict["classifier"].append(classifier)

                if train_test_indexes is None:
                    res_metrics = self.train_and_predict(df, target, classifier, self.metric_list, norm,
                                                            evaluation_type=evaluation_type, test_size=test_size, cv=cv, balancing=balancing)
                else:
                    res_metrics = self.train_and_predict_pro(df, target, classifier, self.metric_list, norm, balancing=balancing, 
                                                            train_test_indexes=train_test_indexes)
                for k in res_metrics:
                    result_dict[k].append(res_metrics[k])

        df = pd.DataFrame(result_dict)
        if melt:
            # return melted df with columns: dataset, classifier, Metric, Score
            # otherwise, returns with columns: dataset, classifier, Accuracy, Recall, ...
            df = df.melt(id_vars=["dataset", "classifier"], value_name="Score", var_name="Metric")
        return df


    def train_and_predict_pro(self, df: pd.DataFrame, target: str, classifier: str, metrics: List[str], norm: bool,
                                balancing: bool, train_test_indexes = None) -> dict:
        # TODO: fix metrics dict

        #print('Running myevaluator.DatasetEvaluator.train_and_predict_PRO')
        features = df.drop(columns=[target]).columns.tolist()
        result = {}
        clf = self.classifiers_dict[classifier]

        train_df = df.loc[train_test_indexes["train"]]
        test_df = df.loc[train_test_indexes["test"]]

        if norm:
            scaler = StandardScaler().fit(train_df[features])
            train_df[features] = scaler.transform(train_df[features])
            test_df[features] = scaler.transform(test_df[features])

        if balancing: 
            train_df = self.balance_data_pro(train_df, target)


        clf.fit(train_df[features].values, train_df[target].values)
        # pred = clf.predict(x_test)

        # cycle for input metrics
        for metric in metrics:
            m_func = self.metrics_dict[metric]
            res = m_func(clf, test_df[features].values, test_df[target].values)
            # complete row with metrics
            result[metric] = res

        return result

    @memoize(table_name='myevaluator_train_and_predict', ignore_self=True)
    def train_and_predict(self, df: pd.DataFrame, target: str, classifier: str, metrics: List[str], norm: bool,
                          evaluation_type: str, test_size: float, cv: int, balancing: bool, train_test_indexes = None) -> dict:
        # TODO: fix metrics dict

        #print('Running myevaluator.DatasetEvaluator.train_and_predict')
        result = {}
        clf = self.classifiers_dict[classifier]

        y = df[target].values
        x = df.drop(columns=[target]).values
        scaled_x = StandardScaler().fit_transform(x) if norm else x
        if evaluation_type == 'holdout':
            if train_test_indexes is None:
                x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=test_size,
                                                                    random_state=RANDOM_STATE)
            else:
                print(x)

            if balancing: 
                x_train, y_train = self.balance_data(x_train, y_train)

            clf.fit(x_train, y_train)
            # pred = clf.predict(x_test)

            # cycle for input metrics
            for metric in metrics:
                m_func = self.metrics_dict[metric]
                res = m_func(clf, x_test, y_test)
                # complete row with metrics
                result[metric] = res
        elif evaluation_type == 'cross-validation' or evaluation_type == 'cv':
            # scoring = {key: self.metrics_dict[key] for key in metrics}  # get dict with only the metrics to use
            # scoring = ['precision_macro', 'recall_macro']
            # scores = cross_validate(clf, scaled_x, y, scoring=scoring, cv=cv)
            cv_result = {key: [] for key in metrics}
            skf = StratifiedKFold(n_splits=cv, random_state=RANDOM_STATE, shuffle=True)
            for i, (train_index, test_index) in enumerate(skf.split(scaled_x, y)):
                x_train, y_train = scaled_x[train_index], y[train_index]
                x_test, y_test = scaled_x[test_index], y[test_index]
                if balancing: 
                    x_train, y_train = self.balance_data(x_train, y_train)
                clf = sklearn.base.clone(clf)   # copy of classifier that has not been fitted
                clf.fit(x_train, y_train)
                for metric in metrics:
                    m_func = self.metrics_dict[metric]
                    res = m_func(clf, x_test, y_test)
                    # complete row with metrics
                    cv_result[metric].append(res)

            for metric in metrics:
                score = np.mean(cv_result[metric])
                result[metric] = score

        else:
            raise ValueError(f'evaluation_type "{evaluation_type}" not supported.')
        return result


    @staticmethod
    def balance_data(x, y, balancing_type='smote'):
        if balancing_type == "smote":
            sampling_strategy = 0.8 if len(np.unique(y)) == 2 else 'not majority'
            # print(sampling_strategy)
            balancer = imblearn.over_sampling.SMOTE(random_state = RANDOM_STATE, sampling_strategy=sampling_strategy)
            x_bal, y_bal = balancer.fit_resample(x, y)
        else:
            raise ValueError(f'Balancing {balancing_type} not supported.')
        return x_bal, y_bal
    

    @staticmethod
    def balance_data_pro(df: pd.DataFrame, target: str, balancing_type='smote'):
        # LOSES INDEX
        features = df.drop(columns=[target]).columns.tolist()

        if balancing_type == "smote":
            sampling_strategy = 0.8 if len(np.unique(df[target])) == 2 else 'not majority'
            # print(sampling_strategy)
            balancer = imblearn.over_sampling.SMOTE(random_state = RANDOM_STATE, sampling_strategy=sampling_strategy)
            x_balanced, y_balanced = balancer.fit_resample(df[features].values, df[target].values)
            n_new_records = x_balanced.shape[0] - df.shape[0]
            new_indexes = [f"syntethic_{i}" for i in range(0, n_new_records)]
            new_indexes = df.index.tolist() + new_indexes

            df_balanced = pd.DataFrame(x_balanced, columns=features, index=new_indexes)
            df_balanced[target] = y_balanced

        else:
            raise ValueError(f'Balancing {balancing_type} not supported.')
        return df_balanced


    @staticmethod
    def plot_results(plot_type: str, data: pd.DataFrame, plot_var: str, x: str, y: str, hue: str, n_cols: int = 3, 
                     ylim=None, invert_xaxis=False, title=None, ci="sd"):
        if plot_type not in DatasetEvaluator.plot_types.keys():
            raise ValueError(f"Plot type {plot_type} not supported. Must be one of the following: "
                             f"{list(DatasetEvaluator.plot_types.keys())}")
        else:
            plot_func = DatasetEvaluator.plot_types[plot_type]

        total_n_plots = len(data[plot_var].unique())
        n_rows = math.ceil(total_n_plots / n_cols)
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8 * n_cols, n_rows * 6), tight_layout=True,
                                 sharey='row')
        # fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
        #                          gridspec_kw={'width_ratios': [2,2,1,1], 'height_ratios': [1,]*n_rows},
        #                          constrained_layout=True,
        #                          sharey='row')
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            print(axes)
            axes = [axes]

        if title is not None:
            fig.suptitle(title, fontsize='x-large')
        for i, plot_title in enumerate(data[plot_var].unique()):
            d = data[data[plot_var] == plot_title]
            plot_func(data=d, x=x, y=y, hue=hue, ax=axes[i], ci=ci)
            axes[i].legend(ncol=len(data[hue].unique()))
            axes[i].set_title(plot_title)
            if ylim is not None: axes[i].set_ylim(ylim)
            if invert_xaxis: axes[i].invert_xaxis()
        plt.show()

    @staticmethod
    def plot_results_with_dimensionalities(plot_type: str, data: pd.DataFrame, dimensionalities: pd.DataFrame, plot_var: str, x: str, y: str, hue: str, n_cols: int = 3, 
                     ylim=None, invert_xaxis=False, title=None, ci="sd"):
        if plot_type not in DatasetEvaluator.plot_types.keys():
            raise ValueError(f"Plot type {plot_type} not supported. Must be one of the following: "
                             f"{list(DatasetEvaluator.plot_types.keys())}")
        else:
            plot_func = DatasetEvaluator.plot_types[plot_type]

        total_n_plots = len(data[plot_var].unique()) + 1
        n_rows = math.ceil(total_n_plots / n_cols)
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8 * n_cols, n_rows * 6), tight_layout=True)
        # fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
        #                          gridspec_kw={'width_ratios': [2,2,1,1], 'height_ratios': [1,]*n_rows},
        #                          constrained_layout=True,
        #                          sharey='row')
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            print(axes)
            axes = [axes]

        if title is not None:
            fig.suptitle(title, fontsize='x-large')

        # dimensions plot
        print(dimensionalities)
        sns.barplot(data=dimensionalities, x="dataset", y="dimensions", ax=axes[0])
        axes[0].set_title("Dataset Dimensionalities")

        for i, plot_title in enumerate(data[plot_var].unique(), start=1):
            d = data[data[plot_var] == plot_title]
            plot_func(data=d, x=x, y=y, hue=hue, ax=axes[i], ci=ci)
            axes[i].legend(ncol=len(data[hue].unique()))
            axes[i].set_title(plot_title)
            if ylim is not None: axes[i].set_ylim(ylim)
            if invert_xaxis: axes[i].invert_xaxis()
        plt.show()

    @staticmethod
    def plot_results_3d(plot_type: str, data: pd.DataFrame, row_var: str, col_var: str, x: str, y: str, z: str):
        n_rows = len(data[row_var].unique())
        n_cols = len(data[col_var].unique())
        # fig = plt.figure(figsize=plt.figaspect(0.2))
        fig = plt.figure(figsize=(20, 10), layout="constrained", dpi=100)
        fig.suptitle(f'{x} x {y}')

        ax_counter = 1
        for row in data[row_var].unique():
            for col in data[col_var].unique():
                ax = fig.add_subplot(n_rows, n_cols, ax_counter, projection='3d')
                t_data = data[(data[row_var] == row) & (data[col_var] == col)]
                x_data, y_data, z_data = t_data[x].values, t_data[y].values, t_data[z].values
                if plot_type == 'trisurf':
                    plot = ax.plot_trisurf(x_data, y_data, z_data, cmap='inferno', edgecolor='none')
                elif plot_type == 'scatter':
                    plot = ax.scatter(x_data, y_data, z_data, c=z_data, cmap='viridis', linewidth=0.5)
                else:
                    raise ValueError(f'plot_type "{plot_type}" not supported.')
                ax.set_title(col)
                ax.set_xlabel(x, labelpad=4)
                ax.set_ylabel(y, labelpad=4)
                ax.set_zlabel(z, labelpad=4)
                fig.colorbar(plot, shrink=0.5, aspect=8, location='left')
                ax_counter += 1
        plt.show()
