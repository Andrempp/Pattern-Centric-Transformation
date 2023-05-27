import json
import os
import re
from collections import Counter
import time

import numpy as np
import pandas as pd
from IPython.core.display_functions import display

from packages.pydge import pydge
from sklearn.model_selection import train_test_split

from . import utils

RS = 7

# LOAD/SAVE FROM/TO FILES ##########################################################################################
def save_df(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path)

def load_df(path: str):
    df = pd.read_csv(path, index_col=0)
    return df

def confirm_create_dir(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)

# SPLIT TRAIN AND TEST DATASETS  ############################################################################

def split_train_test(data: str|pd.DataFrame, target: str, test_size = 0.5):
    if isinstance(data, str):
        data = pd.read_csv(data, index_col=0)
    train, test = train_test_split(data, test_size=test_size, stratify=data[target], random_state=RS)
    return train, test


class LayerDataset:
    # TODO: refactor to loose dependability with pydge, operations with deg are done outside this package
    # TODO: loose dependencies with bicpams optimization, should be done outside this package, in myproject maybe


    default_datatype_per_layer = {'mirna': 'rpm', 'mrna': 'tpm', 'protein': 'rppa'}

    @property
    def project_dir(self):
        return f"{self.data_dir}/{self.project}"

    @property
    def layer_dir(self):
        return f"{self.project_dir}/{self.layer}"

    @property
    def layer_file(self):
        return f"{self.layer_dir}/{self.layer}.csv"

    @property
    def counts_file(self):
        return f"{self.layer_dir}/counts.csv"

    @property
    def rpm_file(self):
        return f"{self.layer_dir}/rpm.csv"

    @property
    def tpm_file(self):
        return f"{self.layer_dir}/tpm.csv"

    @property
    def fpkm_file(self):
        return f"{self.layer_dir}/fpkm.csv"
    
    @property
    def rppa_file(self):
        return f"{self.layer_dir}/rppa.csv"
    
    @property
    def clinical_file(self) -> str:
        return f"{self.data_dir}/{self.project}/clinical.csv"
    
    def drop_target_file(self, target: str) -> str:
        return f"{self.data_dir}/{self.project}/targets/drop_{target}.json"
    
    def replace_target_file(self, target: str) -> str:
        return f"{self.data_dir}/{self.project}/targets/replace_{target}.json"
    
    def optimized_params_file(self, target: str, multi_layers: list = None) -> str:
        if multi_layers is None:
            folder = f"{self.layer_dir}/{target}"
        else:
            multi_layer_name = "_".join(sorted(multi_layers))
            pfolder = f"{self.project_dir}/{multi_layer_name}"
            confirm_create_dir(path=pfolder)
            folder = pfolder + f"/{target}"

        confirm_create_dir(path=folder)
        return f"{folder}/optimized_parameters_val.json"
    
    def train_test_indexes_file(self, target: str, multi_layers: list = None) -> str:
        if multi_layers is None:
            folder = f"{self.layer_dir}/{target}"
        else:
            multi_layer_name = "_".join(sorted(multi_layers))
            pfolder = f"{self.project_dir}/{multi_layer_name}"
            confirm_create_dir(path=pfolder)
            folder = pfolder + f"/{target}"

        confirm_create_dir(path=folder)
        return f"{folder}/train_test_indexes.json"





    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, l: str):
        if l not in LayerDataset.default_datatype_per_layer.keys() and l != None:
            raise ValueError(f'Layer "{l}" not valid. Must be on of the following: {list(LayerDataset.default_datatype_per_layer.keys())}')
        self._layer = l


    def __init__(self, data_dir: str, project: str, layer: str = None):
        self.data_dir = data_dir
        self.project = project
        self.layer = layer
        # create directories if necessary, data must already exist
        if not os.path.isdir(self.data_dir):                              
            raise ValueError(f"Data directory '{self.data_dir}' doesn't exist.")
        if not os.path.isdir(self.project_dir):
            print(f"Project directory '{self.project_dir}' doesn't exist.\nCreating...")
            os.mkdir(self.project_dir)
        if layer is not None and not os.path.isdir(self.layer_dir):
            print(f"Layer directory '{self.layer_dir}' doesn't exist.\nCreating...")
            os.mkdir(self.layer_dir)

    
    def check_layer_set(self):
        if self.layer is None:
            raise ValueError("Layer needs to be set for this operation.")
        return 1


    # PATH GETTER #####################################################################################################

    def get_path(self, key: str, target: str = None, pvalue: float = None):
        """Abstraction to construct the path to a given file or dir defined by {key} with the received arguments"""

        self.check_layer_set()

        path_dict = {
            # folders
            "project_dir": f"{self.project_dir}",
            "layer_dir": f"{self.layer_dir}",
            "target_dir": f"{self.layer_dir}/{target}",
            "dge_dir": f"{self.layer_dir}/{target}/dge_{str(pvalue).replace('.', '_')}",
            "test_dir": f"{self.data_dir}/test",
            "global_target_dir": f"{self.project_dir}/targets",
            "drop_target_dir": f"{self.project_dir}/targets",

            # files
            "drop_target_file": f"{self.project_dir}/targets/drop_{target}.json",
            # "ids_file": f"{self.project_dir}/ids.RData",
            "clinical_file": f"{self.clinical_file}",
            "layer_file": f"{self.layer_file}",
            "counts_file": f"{self.counts_file}",
            "rpm_file": f"{self.rpm_file}",
            "tpm_file": f"{self.tpm_file}",
            "fpkm_file": f"{self.fpkm_file}",
            "rppa_file": f"{self.rppa_file}",

            # "counts_target_file": f"{self.layer_dir}/{target}/counts.csv",
            # "rpm_target_file": f"{self.layer_dir}/{target}/rpm.csv",
            # "fpkm_target_file": f"{self.layer_dir}/{target}/fpkm.csv",
            # "tpm_target_file": f"{self.layer_dir}/{target}/tpm.csv",
            "dge_results_file": f"{self.layer_dir}/{target}/dge_pro.rds",
            "filtered_results_file": f"{self.layer_dir}/{target}/filtered.json",
            "dge_data_file": f"{self.layer_dir}/{target}/dge_{str(pvalue).replace('.', '_')}/dge_data.csv",
            "log_data_file": f"{self.layer_dir}/{target}/dge_{str(pvalue).replace('.', '_')}/log_data.csv",
            "optimized_params_file": f"{self.layer_dir}/{target}/optimized_parameters.json",
        }
        try:
            if key == "dge_results_file":
                confirm_create_dir(path_dict["target_dir"])

            return path_dict[key]

            
        except KeyError:
            raise ValueError(
                f"Argument {{key}} '{key}' is not valid, must be one of the following: {list(path_dict.keys())}")


    # NEW-GEN ################################################################################################

    # TRAIN-TEST SPLIT ################################################################################################

    def generate_train_test_indexes(self, target: str, multi_layers: list = None) -> dict:
        # TODO: make different file according to percentage of test and RS used

        if multi_layers is None:
            self.check_layer_set()
            # generate indexes
            default_data_type = self.default_datatype_per_layer[self.layer]
            data = self.get_data_with_target(data_type=default_data_type, target=target)
        else:
            data = self.get_multiomics_dge_data(target, multi_layers, None, filter_only=True, log_transform=True)

        train, test = split_train_test(data=data, target=target, test_size=0.5)
        indexes = {"train": train.index.tolist(), "test": test.index.tolist()}

        # save indexes in file
        indexes_file = self.train_test_indexes_file(target, multi_layers=multi_layers)
        with open(indexes_file, "w") as f:
            json.dump(indexes, f)
        
        return indexes


    def get_train_test_indexes(self, target: str, multi_layers: list = None):
        if multi_layers is None:
            self.check_layer_set()

        indexes_file = self.train_test_indexes_file(target, multi_layers=multi_layers)
        try:
            with open(indexes_file, "rb") as f:
                indexes = json.load(f)
            return indexes
        except FileNotFoundError:
            print("Train and test indexes not found, generating ...")
            indexes = self.generate_train_test_indexes(target, multi_layers)
            return indexes


    # CLINICAL DATA RELATED METHODS ##################################################################################
    def get_clinical_to_drop(self, target: str) -> list:
        to_drop_file = self.drop_target_file(target)
        try:
            with open(to_drop_file, "rb") as f:
                to_drop = json.load(f)
            return to_drop
        except FileNotFoundError:
            raise FileNotFoundError(f"File {to_drop_file} doesn't exist. Create using: {self.set_clinical_to_drop.__name__}")
    

    def get_clinical_to_replace(self, target: str) -> list:
        to_replace_file = self.replace_target_file(target)
        try:
            with open(to_replace_file, "rb") as f:
                to_drop = json.load(f)
            return to_drop
        except FileNotFoundError:
            raise FileNotFoundError(f"File {to_replace_file} doesn't exist. Create using: {self.set_clinical_to_replace.__name__}")


    def set_clinical_to_drop(self, target: str, values_to_drop: list) -> None:
        to_drop_file = self.drop_target_file(target)
        folder = os.path.dirname(to_drop_file)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        with open(to_drop_file, "w") as f:
            json.dump(values_to_drop, f)
        print(f'Created {to_drop_file}')


    def set_clinical_to_replace(self, target: str, values_to_replace: dict) -> None:
        to_replace_file = self.replace_target_file(target)
        folder = os.path.dirname(to_replace_file)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        with open(to_replace_file, "w") as f:
            json.dump(values_to_replace, f)
        print(f'Created {to_replace_file}')


    def get_clinical_data(self) -> pd.DataFrame:
        """Returns a pandas DataFrame with the clinical data for this project"""

        clinical_file = self.clinical_file
        try:
            df = pd.read_csv(clinical_file, index_col=0)
        except FileNotFoundError:
            print(f"Clinical file '{clinical_file}' doesn't exist.\nCreating...")
            utils.generate_clinical(self.project, clinical_file)
            df = pd.read_csv(clinical_file, index_col=0)
        return df


    # OPTIMIZATION OF BICPAMS HYPERPARAMETERS ##################################################################################

    def set_optimized_bic_parameters(self, target, parameterization, multi_layers = None) -> None:
        optimized_file = self.optimized_params_file(target, multi_layers)
        with open(optimized_file, "w") as f:
            json.dump(parameterization, f)


    def get_optimized_bic_parameters(self, target: str, multi_layers: list = None) -> dict:
        optimized_file = self.optimized_params_file(target, multi_layers)
        try:
            with open(optimized_file, "rb") as f:
                parameterization = json.load(f)
            return parameterization
        except FileNotFoundError:
            raise FileNotFoundError(f"File {optimized_file} doesn't exist. Create using: {self.set_optimized_bic_parameters.__name__}")
    

    # RAW DATA #########################################################################################################
    def get_raw_data(self, data_type: str) -> pd.DataFrame:
        file = self.get_path(f"{data_type}_file")
        df = pd.read_csv(file, index_col=0)
        return df

    def set_raw_data(self, data_type: str, df: pd.DataFrame) -> None:
        file = self.get_path(f"{data_type}_file")
        df.to_csv(file)


    # DGE DATA #########################################################################################################
    def get_log_data(self, target:str, data_type='default') -> pd.DataFrame:
        if data_type=='default':
            data_type = LayerDataset.default_datatype_per_layer[self.layer]
        df = self.get_data_with_target(data_type=data_type, target=target)
        log_df = self.log_transform_data(df, target, data_type)
        return log_df
    

    def get_dge_data(self, target: str, pvalue: float, filter_only=False, data_type='default', log_transform=True) -> pd.DataFrame:
        if data_type=='default':
            data_type = LayerDataset.default_datatype_per_layer[self.layer]

        #TODO: change to individual function for file; after this, need to split dataset into train in another place
        dge_file = self.get_path("dge_results_file", target)
        if not os.path.isfile(dge_file):
            counts_df = self.get_data_with_target(data_type='counts', target=target)
            # filter to only train <----
            train_test_indexes = self.get_train_test_indexes(target)
            print("counts before: ", counts_df.shape)
            counts_df = counts_df.loc[train_test_indexes["train"], :]
            print("counts after: ", counts_df.shape)
            pydge.generate_dge_file(counts_df, target, dge_file)
            del counts_df

        layer_data = self.get_data_with_target(data_type=data_type, target=target)
        df = pydge.get_dge_data(layer_data, target, dge_file, pvalue_cutoff=pvalue, filter_only=filter_only)
        if log_transform:
            df = self.log_transform_data(df, target, data_type)

        return df
    

    def log_transform_data(self, df: pd.DataFrame, target: str, data_type: str) -> pd.DataFrame:
        if data_type=='default':
            data_type = LayerDataset.default_datatype_per_layer[self.layer]
    
        target_col = df[target]
        df = df.drop(columns=[target])

        if data_type == 'rpm':
            total_counts_per_sample = self.get_raw_data("counts").sum(axis="columns")
            df, total_counts_per_sample = utils.match_id_levels(df, total_counts_per_sample)
            total_counts_per_sample = (10 ** 6) / total_counts_per_sample
            for ind, row in df.iterrows():
                df.loc[ind, :] += total_counts_per_sample[ind]
        elif data_type == 'tpm':
            counts_df = self.get_raw_data("counts")
            df, counts_df = utils.match_id_levels(df, counts_df)
            counts_df = counts_df.loc[df.index, df.columns]
            to_add = df / counts_df
            to_add = to_add.fillna(to_add.mean().mean())  # imputation for cases with 0 counts, fill with mean
            df = df + to_add
        else:
            raise ValueError(f"Function 'log_transform_data' not implemented for data type: {data_type}")

        df = np.log2(df)
        df[target] = target_col
        return df

    # MULTIOMICS DATA #############################################################################################
    
    def get_multiomics_dge_data(self, target, layers, pvalue, filter_only=False, data_type="default", log_transform=True) -> pd.DataFrame:
        original_layer = self.layer # save current layer to reset at the end
        dge_datasets = []
        for layer in layers:
            self.layer = layer
            if layer == "protein":
                data = self.get_data_with_target(data_type='rppa', target=target)
            else:
                data = self.get_dge_data(target, pvalue, filter_only, data_type, log_transform)
            print(f"{layer} shape: {data.shape} ")
            dge_datasets.append(data)
        self.layer = original_layer

        df_multi = pd.concat(dge_datasets, axis='columns', join='inner')
        del dge_datasets
        df_multi = df_multi.loc[:,~df_multi.columns.duplicated()] # drop duplicated target column
        # more target column to the end
        target_column = df_multi.pop(target)
        df_multi.insert(len(df_multi.columns), target, target_column)
        print(f"multi-omics shape: {df_multi.shape}")
        return df_multi


    # CLINICAL + TARGET DATASETS ################################################################################################
    
    def get_data_with_target(self, data_type: str, target: str):
        if data_type=='default':
            data_type = LayerDataset.default_datatype_per_layer[self.layer]
        df = self.get_raw_data(data_type)
        clinical_df = self.get_clinical_data()

        # getting target column from clinical data
        try:
            target_column = clinical_df[target]
        except KeyError:
            raise ValueError(f"Invalid argument {{target}} '{target}'. Must be one of the following: "
                             f"{list(clinical_df.columns)}")
        
        # droping values in target
        try:
            values_to_drop = self.get_clinical_to_drop(target)
            target_column = target_column[~target_column.isin(values_to_drop)]
        except FileNotFoundError:
            print(f"No file with values to drop. Continuing.")
            pass

        # replacing values in target
        try:
            replace_dict = self.get_clinical_to_replace(target)
            target_column = target_column.replace(replace_dict)
        except FileNotFoundError:
            print(f"No file with values to replace. Continuing.")
            pass

        df, target_column = utils.match_id_levels(df, target_column)
        df = pd.merge(df, target_column, left_index=True, right_index=True)

        # define positive (1) and negative (0) classes in binary, 0, .., n otherwise
        target_values = df[target].value_counts(sort=True, ascending=False).keys()
        target_replace = dict(zip(target_values, range(len(target_values))))
        df[target] = df[target].replace(target_replace)

        return df

    # Get complete datasets for this project ###########################################################################
    def get_layer(self, n_rows: int = None) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.layer_file, nrows=n_rows, index_col=0)
        except FileNotFoundError:
            print(f"Layer file '{self.layer_file}' doesn't exist.\nCreating...")
            self.generate_layer()
            df = pd.read_csv(self.layer_file, nrows=n_rows, index_col=0)
        return df

    def generate_layer(self):
        # allow for the use of a 'test' layer that behaves just like a normal layer until this point where it gets the
        # data equivalent to the mirna layer
        layer = 'mirna' if self.layer == 'test' else self.layer
        utils.generate_layer(self.project, layer,  self.data_dir, self.layer_file)

    # Get information about columns of layer ###########################################################################
    def get_columns_of_layer(self) -> list:
        """Gets the list of columns for this layer"""

        try:
            columns = pd.read_csv(self.layer_file, nrows=0).columns.tolist()
        except FileNotFoundError:
            print(f"Layer file '{self.layer_file}' doesn't exist.\nCreating...")
            self.generate_layer()
            columns = pd.read_csv(self.layer_file, nrows=0).columns.tolist()
        return columns

    def get_types_of_columns(self) -> Counter:
        """Gets types of columns when there are multiple columns for a single patient"""

        cols = self.get_columns_of_layer()
        types = list(map(lambda x: x.split("TCGA")[0], cols))
        return Counter(types)

    def get_columns_of_type(self, type_of_column: str) -> list:
        """Gets the list of columns of a certain type, as returned by 'get_types_of_columns'"""

        columns = self.get_columns_of_layer()
        if type_of_column == "":
            type_of_column = "TCGA"
        r = re.compile(f"{type_of_column}.*")
        columns = list(filter(r.match, columns))
        return columns

    # Get partial datasets for layer ###################################################################################
    def get_layer_by_columns(self, columns: list) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.layer_file, usecols=columns, index_col=0)
        except FileNotFoundError:
            print(f"Layer file '{self.layer_file}' doesn't exist.\nCreating...")
            self.generate_layer()
            df = pd.read_csv(self.layer_file, usecols=columns, index_col=0)
        return df

    def get_layer_by_column_type(self, type_of_column: str, keep_unique_columns=True) -> pd.DataFrame:
        """Gets a pandas DataFrame with only a specified type of column, as obtained by 'get_types_of_columns'."""

        columns = self.get_columns_of_type(type_of_column)

        if keep_unique_columns:
            counter_columns = self.get_types_of_columns()
            unique_columns = [item[0] for item in counter_columns.items() if item[1] == 1]
            columns = columns + unique_columns

        df = self.get_layer_by_columns(columns)
        df.columns = [col.replace(type_of_column, "") for col in df.columns]
        return df

    def get_layer_by_sample(self, sample_id: str, keep_unique_columns=True) -> pd.DataFrame:
        # TODO: receive list of samples instead of a single one.
        # TODO: Speed up selection of matching columns
        columns = self.get_columns_of_layer()
        columns = [col for col in columns if sample_id in col]
        if keep_unique_columns:
            counter_columns = self.get_types_of_columns()
            unique_columns = [item[0] for item in counter_columns.items() if item[1] == 1]
            columns = columns + unique_columns

        df = self.get_layer_by_columns(columns)
        return df


    def __str__(self):
        pass

    def __repr__(self):
        pass
