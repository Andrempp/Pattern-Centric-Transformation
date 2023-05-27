import os

import pandas as pd
import numpy as np
import sys
import time
import subprocess
from typing import Union, List
import glob
import re


MODULE_DIR = os.path.dirname(__file__)
R_SCRIPT_DIR = os.path.join(MODULE_DIR, 'r_dir/')
TEMP_ID_FILE = os.path.join(MODULE_DIR, 'temp_ids.RData')

# Utils ###############################################################################
# Utility functions ###################################################################
#######################################################################################

def execute(cmd):
    """Creates subprocess to run received command and yields the output as it is produced"""

    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def get_id_column(df: pd.DataFrame) -> str:
    """Finds the ID column of a DataFrame by searching for the column with unique values for all rows"""
    
    nrows = df.shape[0]
    for column in df.columns:
        if len(df[column].value_counts()) == nrows:
            return column
    
    # if reaches this, then no column with only IDs
    # select column with most unique values
    nu = df.nunique().idxmax()
    return nu
    #raise RuntimeError("No id column found") 


def reduce_numeric_64_to_32(df: pd.DataFrame, verbose = 0) -> pd.DataFrame:
    """Reduces int64 and float64 columns to int32 and float32 if possible"""

    max_int32 = np.iinfo(np.int32).max
    max_float32 = np.finfo(np.float32).max
    counter_int = 0
    counter_float = 0
    for col in df.columns:
        coltype = df[col].dtype
        if coltype == np.int64:
            if (df[col].abs() < max_int32).all():
                df[col] = df[col].astype("int32")
                counter_int +=1
        elif coltype == np.float64:
            if (df[col].abs() < max_float32).all():
                df[col] = df[col].astype("float32")
                counter_float+=1
                
    if verbose >= 1: print(f"Reduced {counter_int} ints and {counter_float} floats")
    return df

# Generate Functions ##################################################################
# Used to run R scripts that download the data from TCGA and store it in disk #########
#######################################################################################

def generate_ids(project: str, ids_file_path: str, verbose: int = 1) -> None:
    """Generates the ids.RData file for the corresponding project and stores it in disk"""

    print(["Rscript", f'{R_SCRIPT_DIR}get_ids.R', project, ids_file_path])
    for output in execute(["Rscript", f'{R_SCRIPT_DIR}get_ids.R', project, ids_file_path]):
        if verbose >= 1: print(output, end="")


def generate_clinical(project: str, clinical_file_path: str, verbose: int = 1) -> None:
    """Generates a processed dataframe with clinical data from the data of TCGA and stores it in disk"""

    for output in execute(["Rscript", f'{R_SCRIPT_DIR}get_clinical.R', project, clinical_file_path]):
        if verbose >= 1: print(output, end="")


def generate_layer(project: str, layer: str, data_dir: str, layer_file: str, divider: int = -1,
                   verbose: int = 1) -> None:
    """Generates a processed dataframe (or multiple) from the data of TCGA and stores it in disk"""

    try:
        generate_ids(project, TEMP_ID_FILE, verbose=verbose)    # create temp file with layer IDs for get_omics.R
        for output in execute(
                ["Rscript", f'{R_SCRIPT_DIR}get_omics.R', project, layer, "TRUE", str(divider), data_dir, layer_file, TEMP_ID_FILE]):
            if verbose >= 1: print(output, end="")
    finally:
        #os.remove(TEMP_ID_FILE)  # delete temp file with layer IDs
        print()
    merge_segments_of_layer(layer, layer_file, verbose=verbose)  # Merges the segments and stores complete layer in disk


##### Get Functions ##########################################################################


# def get_indexes_of_layer(project: str, layer: str, data_dir: str) -> list:
#     """Gets the list of indexes for a specific omics layer"""
#
#     layer = layer.lower()
#     project = project.upper()
#     layer_file = get_path("layer_file", data_dir, project, layer)
#     try:
#         indexes = pd.read_csv(layer_file, usecols=[0])
#     except FileNotFoundError:
#         print(f"File not found, creating dataset for project '{project}' and layer '{layer}'")
#         generate_layer(project, layer, data_dir)
#         indexes = pd.read_csv(layer_file, usecols=[0])
#     id_col = indexes.columns.tolist()[0]
#     return indexes[id_col].tolist()
#
#
# def get_ids_in_layer(project: str, layer: str, data_dir: str, level_of_barcode: str = "default"):
#     """Gets the list of ids for a specific omics layer. Even if there are multiple types of column. Can select the
#     level of the barcode id """
#
#     possible_levels_of_id = {"default": -1, "participant": 3, "sample": 4, "aliquot": 7}
#
#     if level_of_barcode not in possible_levels_of_id:
#         raise ValueError(
#             f"Argument 'level_of_barcode' with invalid value: '{level_of_barcode}'\nValid values: {possible_levels_of_id}")
#
#     types_counter = get_types_of_columns(project, layer, data_dir)
#     most_common_type = types_counter.most_common(1)[0][0]
#
#     columns = get_columns_of_type(project, layer, data_dir, most_common_type)
#
#     ids = [col.replace(most_common_type, "") for col in columns]
#
#     cut_point = possible_levels_of_id[level_of_barcode]
#
#     if len(ids[0].split("-")) < cut_point:
#         raise ValueError(f"Argument 'level_of_barcode' too wide for available data: '{level_of_barcode}'\nExample of "
#                          f"available id: {ids[0]}")
#
#     ids = ['-'.join(iid.split("-")[:cut_point]) for iid in ids]
#
#     return ids


### Functions to get partial datasets ############################################################################

# def get_layer_by_column_type(project: str, layer: str, data_dir: str, type_of_column: str,
#                              keep_unique_columns=True) -> pd.DataFrame:
#     """Gets a pandas DataFrame with a specific type of column, as obtained by 'get_types_of_columns'."""
#
#     columns = get_columns_of_type(project, layer, data_dir, type_of_column)
#
#     if keep_unique_columns:
#         counter_columns = get_types_of_columns(project, layer, data_dir)
#         unique_columns = [item[0] for item in counter_columns.items() if item[1] == 1]
#         columns = columns + unique_columns
#
#     df = get_layer_divided_vertically(project, layer, data_dir, columns=columns)
#     df.columns = [col.replace(type_of_column, "") for col in df.columns]
#
#     return df


# def get_layer_nrows(project: str, layer: str, data_dir: str, nrows: int = -1) -> pd.DataFrame:
#     layer = layer.lower()
#     project = project.upper()
#     layer_file = get_path("layer_file", data_dir, project, layer)
#     if nrows == -1: nrows = None
#
#     try:
#         df = pd.read_csv(layer_file, nrows=nrows)
#     except FileNotFoundError:
#         print(f"File not found, creating dataset for project '{project}' and layer '{layer}'")
#         generate_layer(project, layer, data_dir)
#         df = pd.read_csv(layer_file, nrows=nrows)
#     return df


# def get_layer_divided_vertically(project: str, layer: str, data_dir: str,
#                                  columns: Union[list, int] = -1) -> pd.DataFrame:
#     """Gets a pandas DataFrame for a specific omics layer divided vertically (by columns)"""
#
#     layer = layer.lower()
#     project = project.upper()
#     layer_file = get_path("layer_file", data_dir, project, layer)
#     try:
#         if columns == -1:
#             columns = get_columns_of_layer(project, layer, data_dir)
#         df = pd.read_csv(layer_file, usecols=columns)
#     except FileNotFoundError:
#         print(f"File not found, creating dataset for project '{project}' and layer '{layer}'")
#         generate_layer(project, layer, data_dir)
#         if columns == -1:
#             columns = get_columns_of_layer(project, layer, data_dir)
#         df = pd.read_csv(layer_file, usecols=columns)
#     return df
#
#
# def get_layer_divided_horizontally(project: str, layer: str, data_dir: str, divider: int) -> pd.DataFrame:
#     """Gets a pandas DataFrame for a specific omics layer divided horizontaly (by rows) !generator!"""
#
#     layer = layer.lower()
#     project = project.upper()
#     n_indexes = len(get_indexes_of_layer(project, layer, data_dir))
#     total_lines = range(0, n_indexes + 1)
#     segments = np.array_split(total_lines[1:], divider)
#     layer_file = get_path("layer_file", data_dir, project, layer)
#     try:
#         df = pd.read_csv(layer_file, nrows=0)
#     except FileNotFoundError:
#         print(f"File not found, creating dataset for project '{project}' and layer '{layer}'")
#         generate_layer(project, layer, data_dir)
#
#     for i in range(0, divider):
#         a = time.time()
#         # to_skip = [j for j in total_lines[1:] if j not in segments[i]]
#         to_skip = list(set(total_lines[1:]) - set(segments[i]))
#         df = pd.read_csv(layer_file, skiprows=to_skip, header=0)
#         yield df


##### Transform Functions ##########################################################################
# Apply transformations to the already loaded data #################################################
####################################################################################################

# def add_clinical_target(project: str, data_dir: str, df: pd.DataFrame, clinical_target: str) -> pd.DataFrame:
#     clinical_df = get_clinical(project, data_dir).set_index("submitter_id")  # TODO: hardcoded index column name
#     target = clinical_df[clinical_target]
#
#     df, target = match_id_levels(df, target)
#     df = pd.merge(df, target, left_index=True, right_index=True)
#     return df


def match_id_levels(df1: pd.DataFrame, df2: pd.DataFrame, deal_with_duplicates="delete") -> List[pd.DataFrame]:
    """Matches ID levels of two dataframes by reducing the longest type of ID to match the shortest"""
    # TODO: change to receive list of dataframes

    valid_deal_with_duplicates = ["delete"]

    df_list = [df1, df2]
    id1 = df_list[0].index[0].split("-")
    id2 = df_list[1].index[0].split("-")

    len_list = [len(id1), len(id2)]
    min_index = np.argmin(len_list)
    reduce_index = 1 - min_index

    new_index = ["-".join(i.split("-")[:len_list[min_index]]) for i in df_list[reduce_index].index]

    df_list[reduce_index].index = new_index

    if deal_with_duplicates == "delete":
        df_list[reduce_index] = df_list[reduce_index][~df_list[reduce_index].index.duplicated(keep='first')]
    else:
        raise ValueError(
            f"Invalid deal_with_duplicates argument '{deal_with_duplicates}'. Should be one of the following: {valid_deal_with_duplicates}.")

    return df_list


def merge_segments_of_layer(layer: str, layer_file: str, verbose=0) -> None:
    """Merges the segments of the layer dataset as produced by 'generate_layer' into a complete layer dataset"""
    # TODO: functions is doing more than one thing, merges AND fixes CNV columns names.
    #  Layer arg is only used for the latter

    layer = layer.lower()
    mb_size = 1024 ** 2

    search_string = layer_file.replace('.csv', '_*.csv')
    print(search_string)
    segments_files = sorted(glob.glob(search_string))
    if len(segments_files) == 0:
        raise FileNotFoundError(f"No files of type '{search_string}' found.")

    final = pd.read_csv(segments_files[0])
    final = reduce_numeric_64_to_32(final, verbose=1)

    id_col = get_id_column(final)

    for i, file in enumerate(segments_files[1:]):
        if verbose >= 1:
            print(
                f"Segment {i + 2} out of {len(segments_files)}\t\tCurrent size of dataframe: {round(sys.getsizeof(final) / mb_size)} MB",
                end='\r')
        t = pd.read_csv(file)
        t = reduce_numeric_64_to_32(t, verbose=1)

        diff_cols = t.columns.difference(final.columns).tolist() + [id_col]
        final = final.merge(t.loc[:, diff_cols], how="inner", on=[id_col])
        del t

    # correct unuseful column names of cnv data
    if layer == "cnv":
        r = re.compile(f"TCGA.*")
        columns = list(filter(r.match, final.columns))
        new_columns = ["_".join(c.split('_')[1:]) + '_' + c.split(',')[0] for c in columns]
        replace_dict = dict(zip(columns, new_columns))
        final.rename(columns=replace_dict, inplace=True)

    final.to_csv(layer_file, index=False)
    if verbose >= 1:
        print(f'File: "{layer_file}" saved successfully')


